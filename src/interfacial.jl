# Step 3: 1D interfacial profiles, interfacial tension, and electrostatic
# potential profiles along a sequence's Step 1 binodal.
#
# The naive, uncorrected periodic double-interface ("slab") tanh initial
# guess this file used to build locally (`initialize_twophase_profile`) is
# now `ClassicalDFT.initialize_profiles(sys)` itself: ClassicalDFT dispatches
# on `model::ElectrolyteModel` (`ClassicalDFT/src/structure/two_phase.jl`)
# to build exactly this per-component periodic tanh shape (fixed
# `width_frac=0.05`, matching this project's own former default) instead of
# its generic `Clapeyron.split_pure_model`+`crit_pure`-based `cos_prof`
# method, which fails for LS's degenerate single-group ion components. The
# domain-integrated electroneutrality Boltzmann rescale is applied
# automatically by that same entry point for any system with an
# `ElectrostaticPotentialModel` external field -- no separate correction
# call needed here anymore. A genuinely net-charged system's bulk-to-bulk
# Donnan potential jump (Ψ≠0) is likewise imposed automatically now, via
# `ClassicalDFT.impose_donnan_structure!` (called internally by
# `ClassicalDFT.initialize_profiles(sys)` for any `ElectrolyteModel` two-phase
# system) -- reading Ψ directly off the bulk phases already stored on
# `sys.structure` (`Clapeyron.donnan_psi_bulk`), with no caller involvement;
# a no-op when the bulk phases are already Donnan-neutral (e.g. the
# neutral-chain+symmetric-salt production systems, `Ψ≡0` by symmetry).

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
    sys = ClassicalDFT.ElectrolyteDFTSystem(model, structure)
    rho = ClassicalDFT.initialize_profiles(sys)

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
    run_interfacial_point_charged(model, T, x_dense, y, V_dense, V_dilute;
                                   ngrid=(201,), halfbox_L=30.0, maxit=5000, tol=1e-6, anderson_m=5)

Multi-component analogue of `run_interfacial_point`, for a genuine
polyampholyte+counter-ion (or more generally N-component) `LS` model. Takes
the converged output of
`Clapeyron.bubble_pressure(model, T, x_dense, DonnanBubblePressure())`
directly (`x_dense`/`y`/`V_dense`/`V_dilute`) as the two bulk phases'
boundary condition, rather than re-deriving bulk densities from a single
scalar star-density (ambiguous once there is more than one species) the way
`run_interfacial_point` does for the single-component case.

Returns the same `(gamma, x, rho, Vext)` shape as `run_interfacial_point`;
`rho`'s last dimension spans every bead across every component (chain beads
then ion bead(s), matching `model.charge`'s own ordering).
"""
function run_interfacial_point_charged(model::LS, T::Real, x_dense::AbstractVector, y::AbstractVector,
                                        V_dense::Real, V_dilute::Real;
                                        ngrid=(201,), halfbox_L::Float64=30.0,
                                        maxit::Int=5000, tol::Float64=1e-6, anderson_m::Int=5,
                                        verbose::Bool=true)
    L = ClassicalDFT.length_scale(model)

    rhobulk_dense = collect(x_dense) ./ V_dense
    rhobulk_dilute = collect(y) ./ V_dilute
    p_coex = Clapeyron.pressure(model, V_dense, T, x_dense)

    structure = ClassicalDFT.TwoPhase1DCart((p_coex, T), rhobulk_dense, rhobulk_dilute,
                                             [-halfbox_L * L, halfbox_L * L], ngrid)
    sys = ClassicalDFT.ElectrolyteDFTSystem(model, structure)
    rho = ClassicalDFT.initialize_profiles(sys)

    if verbose
        ratio = sum(rhobulk_dense) / sum(rhobulk_dilute)
        println("  T=$(round(T, digits=4))  ratio=$(round(ratio, sigdigits=3))...")
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
`ClassicalDFT.surface_tension`'s raw two-interface value, since the periodic
box has two interfaces), the shared grid `x` (units of
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
        sys = ClassicalDFT.ElectrolyteDFTSystem(model, structure)
        rho = ClassicalDFT.initialize_profiles(sys)

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
