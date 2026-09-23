# Step 3: 1D interfacial profiles, interfacial tension, and electrostatic
# potential profiles along a sequence's Step 1 binodal.

"""
    initialize_twophase_profile(sys::LSDFTSystem; width_frac=0.05)

Builds a **periodic** double-interface ("slab") initial density profile
between the two bulk phases stored in `sys.structure` (`ρbulk`/
`topology.ρbulk2`), one identical copy per bead (every bead shares the same
per-chain bulk density in the uniform limit -- see `LSDFTNeutral.jl`/
`LSDFTIon.jl`). Phase 1 (`ρbulk`) occupies the middle half of the box,
phase 2 (`ρbulk2`) the two outer quarters -- which are contiguous once the
box wraps periodically, so the profile is continuous (and smooth) all the
way around, with two symmetric interfaces.

This periodicity is not optional: `ClassicalDFT`'s `evaluate_field!`/
`propagate!` machinery is FFT-based and inherently periodic. An earlier
version of this function used `ClassicalDFT.tanh_prof` for a single,
one-sided transition (`ρbulk` at the left edge, `ρbulk2` at the right) --
which is *discontinuous* once the box wraps from `ub` back to `lb`. That
discontinuity gets smoothed away by the weighted-density convolutions within
the first few iterations, and `converge!` reports a clean, fully-converged
residual for a spurious uniform intermediate density instead of the
intended two-phase profile -- a silent failure mode, not an error.

Mirrors `ClassicalDFT.initialize_profiles`'s own `TwoPhaseSystem{:Cartesian}`
method (`ClassicalDFT/src/structure/two_phase.jl`, which uses the
inherently-periodic `cos_prof` for the same reason) rather than reusing it
directly, because that method also calls `Clapeyron.crit_pure(pure_model)`
with no custom initial guess, purely to pick a cosmetic steepness for the
cosine shape. `Clapeyron`'s generic `crit_pure` guess assumes a van-der-
Waals/SAFT-like critical packing fraction (~0.13-0.3) -- badly wrong for
this EOS's actual ~0.01-0.02 (the same issue Step 1's `find_critical_point`
was built to work around via bisection + a properly-scaled guess, see
`phasediagram.jl`), and the trust-region solver can spin for a very long
time rather than failing fast from a guess that far off. Since that call
only shapes the *cosmetic* initial guess (not anything `converge!`'s
correctness depends on), building an equally-reasonable periodic profile
directly with two plain `tanh` steps -- `width_frac` controls each
interface's width as a fraction of the box -- avoids it entirely without
needing a parallel `crit_pure` fix.

Because there are two interfaces, downstream `ClassicalDFT.surface_tension`
values must be halved to get the physical (single-interface) tension --
exactly the convention `ClassicalDFT.surface_tension(model,T,x)`'s own
convenience method already uses.
"""
function initialize_twophase_profile(sys::LSDFTSystem; width_frac::Float64=0.05)
    structure = sys.structure
    ngrid = structure.ngrid
    ρ1 = structure.ρbulk[1]         # phase 1 -- middle half of the box
    ρ2 = structure.topology.ρbulk2[1] # phase 2 -- outer quarters
    lb, ub = ClassicalDFT.bounds(structure, 1)
    H = ub - lb
    x = collect(ClassicalDFT.uniform_range(structure, 1))

    # s ∈ [0,1): interface 1 (ρ2→ρ1) at s=0.25, interface 2 (ρ1→ρ2) at s=0.75.
    s = @. (x - lb) / H
    ρ_points = @. ρ2 + (ρ1 - ρ2) * 0.5 * (tanh((s - 0.25) / width_frac) - tanh((s - 0.75) / width_frac))

    nbeads = sum(sys.species.nbeads)
    FP = ClassicalDFT.fptype(sys.options)
    device = sys.options.device
    ρ = ClassicalDFT.allocate(device, FP, ngrid..., nbeads)
    for j in 1:nbeads
        ρ[:, j] = ClassicalDFT.adapt_to_device(device, FP, ρ_points)
    end
    return ρ
end

"""
    select_binodal_indices(pd; npoints=5, ratio_max=1e12)

Pick up to `npoints` indices into a `phase_diagram`/`load_phase_diagram`
result `pd`, log-spaced in density ratio `rho_dense/rho_dilute` between 10
and `ratio_max`. `ratio_max` defaults to `1e12` (effectively the full
binodal `phase_diagram` traces for this project's sequences, whose density
ratio tops out around `1e11-1e12` well away from the critical point) --
raised from an earlier, more conservative `1e8` after directly confirming
`run_interfacial_point` still converges cleanly (positive `gamma`, correct
bulk plateaus) at the actual maximum available ratio for a test sequence
(`~7.9e11`, `block5`), not just comfortably below it. Factored out of
`run_interfacial_sequence` so a large point count can be split across
independent parallel (e.g. SLURM array) tasks, each computing one point by
indexing into this same deterministic list, rather than one task looping
over all of them serially.
"""
function select_binodal_indices(pd; npoints::Int=5, ratio_max::Float64=1e12)
    ratio = pd.rho_dense ./ pd.rho_dilute
    valid = findall(r -> 10 <= r <= ratio_max, ratio)
    isempty(valid) && throw(ArgumentError("no binodal points with density ratio in [10, $ratio_max]"))
    logr = log.(ratio[valid])
    targets = range(minimum(logr), maximum(logr), length=min(npoints, length(valid)))
    return unique([valid[argmin(abs.(logr .- t))] for t in targets])
end

"""
    run_interfacial_point(model, lB, rho_dense_star, rho_dilute_star; ngrid=(201,),
                           halfbox_L=30.0, maxit=5000, tol=1e-6, anderson_m=5)

Run a single 1D interfacial `converge!` calculation for an already-built
`expand=true` `LS` model at one known coexistence point (`lB`,
`rho_dense_star`, `rho_dilute_star` -- reduced total-bead densities, the
same convention `phase_diagram`/`find_coexistence` use). Factored out of
`run_interfacial_sequence`'s per-point body so a single point can be run
standalone -- e.g. for a large parallel sample of sequences at one fixed
`lB`, where each sequence is its own (SLURM array) job rather than a serial
sweep along a binodal.

Returns a `NamedTuple` with `gamma` (single-interface interfacial tension),
the shared grid `x` (units of `σ`), bead-resolved density profile `rho`
(`ngrid × nbeads`), and electrostatic potential profile `Vext` (`ngrid`,
reduced `eψ/k_BT`).
"""
function run_interfacial_point(model::LS, lB::Real, rho_dense_star::Real, rho_dilute_star::Real;
                                ngrid=(201,), halfbox_L::Float64=30.0,
                                maxit::Int=5000, tol::Float64=1e-6, anderson_m::Int=5,
                                verbose::Bool=true)
    N = length(model.charge)
    sigma = LS_SIGMA
    L = ClassicalDFT.length_scale(model)
    T = lB_to_T(model, lB)

    rhobulk_dense = [rho_dense_star / (Clapeyron.N_A * N * sigma^3)]
    rhobulk_dilute = [rho_dilute_star / (Clapeyron.N_A * N * sigma^3)]
    p_coex = Clapeyron.pressure(model, 1 / rhobulk_dense[1], T, [1.0])

    structure = ClassicalDFT.TwoPhase1DCart((p_coex, T), rhobulk_dense, rhobulk_dilute,
                                             [-halfbox_L * L, halfbox_L * L], ngrid)
    sys = LSDFTSystem(model, structure)
    rho = initialize_twophase_profile(sys)

    if verbose
        println("  lB=$(round(lB, digits=4))  ratio=$(round(rho_dense_star / rho_dilute_star, sigdigits=3))...")
        flush(stdout)
    end
    ClassicalDFT.converge!(sys, rho; verbose=false, maxit=maxit, tol=tol, anderson_m=anderson_m)
    γ = ClassicalDFT.surface_tension(sys, rho) / 2  # halve: two interfaces in the periodic box

    ef = sys.external_field[1]
    nd = ClassicalDFT.dimension(sys)
    CT = ClassicalDFT.transform_eltype(sys.structure, ClassicalDFT.fptype(sys.options))
    Vext = similar(selectdim(rho, nd + 1, 1), CT)
    P, iP = ClassicalDFT.build_transform(sys.structure, Vext, nd, sys.options.device)
    dfdrho_dummy = similar(rho)
    ClassicalDFT.evaluate_external_field!(sys.structure, ef, sys.model, rho, dfdrho_dummy, P, iP, Vext)
    Vext_reduced = real.(Vext) ./ (Clapeyron.k_B * T)

    x = collect(ClassicalDFT.uniform_range(structure, 1)) ./ L

    if verbose
        println("    γ=$(round(γ, sigdigits=4))")
        flush(stdout)
    end

    return (gamma=γ, x=x, rho=Array(rho), Vext=Vext_reduced)
end

"""
    run_interfacial_sequence(charges, pd; npoints=5, ngrid=(201,), halfbox_L=30.0,
                              ratio_max=1e12, maxit=5000, tol=1e-6, anderson_m=5)

Run 1D interfacial `converge!` calculations for `charges` at up to `npoints`
points along its Step 1 binodal `pd` (a `phase_diagram`/`load_phase_diagram`
result), log-spaced in density ratio `rho_dense/rho_dilute` between 10 and
`ratio_max` (see `select_binodal_indices`, which this delegates the point
selection to). `ratio_max=1e12` covers essentially the full binodal this
project's `phase_diagram` traces -- confirmed to converge cleanly (positive
`gamma`, correct bulk plateaus) all the way to the actual maximum available
ratio for a test case (`~7.9e11`), not just comfortably below some more
conservative earlier cutoff.

Returns a `NamedTuple` recording everything needed to reproduce any of the
diagnostic plots built during Step 3 development, not just the scalar IFT:
`charges`, `kappa_seq`, the `lB`/bulk densities actually used, per-point
single-interface interfacial tension `gamma` (halved from
`ClassicalDFT.surface_tension`'s raw two-interface value -- see this file's
`initialize_twophase_profile` docstring), the shared grid `x` (units of
`σ`), and the full bead-resolved density profiles `rho`
(`npoints × ngrid × nbeads`) and electrostatic potential profiles `Vext`
(`npoints × ngrid`, reduced `eψ/k_BT` units).
"""
function run_interfacial_sequence(charges::AbstractVector{<:Integer}, pd;
                                   npoints::Int=5, ngrid=(201,),
                                   halfbox_L::Float64=30.0, ratio_max::Float64=1e12,
                                   maxit::Int=5000, tol::Float64=1e-6, anderson_m::Int=5,
                                   verbose::Bool=true)
    N = length(charges)
    sigma = LS_SIGMA
    model = LS(charges; expand=true)
    L = ClassicalDFT.length_scale(model)

    idxs = select_binodal_indices(pd; npoints=npoints, ratio_max=ratio_max)

    lB_used = Float64[]
    rho_dense_used = Float64[]
    rho_dilute_used = Float64[]
    gamma = Float64[]
    x_shared = nothing
    rho_all = Vector{Matrix{Float64}}()
    Vext_all = Vector{Vector{Float64}}()

    for idx in idxs
        lB = pd.lB[idx]
        rho_dense_star = pd.rho_dense[idx]
        rho_dilute_star = pd.rho_dilute[idx]
        T = lB_to_T(model, lB)

        rhobulk_dense = [rho_dense_star / (Clapeyron.N_A * N * sigma^3)]
        rhobulk_dilute = [rho_dilute_star / (Clapeyron.N_A * N * sigma^3)]
        # ClassicalDFT.surface_tension reads system.structure.conditions[1] directly
        # as the coexistence pressure (its `p*∫(...)` term) -- it is NOT a cosmetic
        # placeholder the way it is for the plain δFδρ_res bulk-limit checks used
        # earlier in Step 2/3 development. Must be the actual coexistence pressure
        # at this (T, rhobulk_dense) state, not an arbitrary constant: found via a
        # negative-surface-tension regression on the near-alternating (block1)
        # sequence, whose true coexistence pressure (~1e5-2e5 Pa at large lB) is
        # close enough to a naive `1e5` placeholder to look plausible only by
        # coincidence at low lB (block5/asym1's true pressures are ~10^2-10^3 Pa).
        p_coex = Clapeyron.pressure(model, 1 / rhobulk_dense[1], T, [1.0])

        structure = ClassicalDFT.TwoPhase1DCart((p_coex, T), rhobulk_dense, rhobulk_dilute,
                                                 [-halfbox_L * L, halfbox_L * L], ngrid)
        sys = LSDFTSystem(model, structure)
        rho = initialize_twophase_profile(sys)

        if verbose
            println("  lB=$(round(lB, digits=4))  ratio=$(round(rho_dense_star / rho_dilute_star, sigdigits=3))...")
            flush(stdout)
        end
        ClassicalDFT.converge!(sys, rho; verbose=false, maxit=maxit, tol=tol, anderson_m=anderson_m)

        γ = ClassicalDFT.surface_tension(sys, rho) / 2  # halve: two interfaces in the periodic box

        ef = sys.external_field[1]
        nd = ClassicalDFT.dimension(sys)
        CT = ClassicalDFT.transform_eltype(sys.structure, ClassicalDFT.fptype(sys.options))
        Vext = similar(selectdim(rho, nd + 1, 1), CT)
        P, iP = ClassicalDFT.build_transform(sys.structure, Vext, nd, sys.options.device)
        dfdrho_dummy = similar(rho)
        ClassicalDFT.evaluate_external_field!(sys.structure, ef, sys.model, rho, dfdrho_dummy, P, iP, Vext)
        Vext_reduced = real.(Vext) ./ (Clapeyron.k_B * T)

        if x_shared === nothing
            x_shared = collect(ClassicalDFT.uniform_range(structure, 1)) ./ L
        end

        push!(lB_used, lB)
        push!(rho_dense_used, rho_dense_star)
        push!(rho_dilute_used, rho_dilute_star)
        push!(gamma, γ)
        push!(rho_all, Array(rho))
        push!(Vext_all, Vext_reduced)

        if verbose
            println("    γ=$(round(γ, sigdigits=4))")
            flush(stdout)
        end
    end

    npts = length(lB_used)
    ngrid1 = ngrid[1]
    rho_arr = Array{Float64}(undef, npts, ngrid1, N)
    Vext_arr = Array{Float64}(undef, npts, ngrid1)
    for i in 1:npts
        rho_arr[i, :, :] = rho_all[i]
        Vext_arr[i, :] = Vext_all[i]
    end

    return (charges=collect(charges), kappa_seq=kappa_seq(charges), lB=lB_used,
            rho_dense=rho_dense_used, rho_dilute=rho_dilute_used, gamma=gamma,
            x=x_shared, rho=rho_arr, Vext=Vext_arr)
end
