#=
This file was originally built around a standalone, reduced-unit re-derivation
of the free energy (`a_bulk`), rather than the actual `LS` Clapeyron model.
That re-derivation silently used the *reference file's own* two-independent-
bead-type ideal-mixing entropy convention (matching resources/Helmholtz_LS.jl's
own `f0`), which is the right entropy term for a system of two independently
diffusing homopolymer species -- but wrong for this project's actual system,
a single covalently-bonded copolymer sequence (one Clapeyron component, one
ideal-gas-of-chains entropy term, exactly what `LSNeutral`/`LSIon` + Clapeyron's
`BasicIdeal` already give correctly via `a_res`). That mismatch produced
spurious phase coexistence at Bjerrum lengths far below the true critical
point (confirmed directly against `Clapeyron.crit_pure` and a from-scratch
`Clapeyron.pressure` scan). This version works entirely through Clapeyron's
own generic (V,T,z) machinery -- `pressure`, `VT_chemical_potential`, `crit_pure`
-- on the real `LS` model, so it can never again drift from what `a_res`
actually encodes.
=#

"""
    lB_to_T(model, lB; z=SA[1.0])

Convert a target (reduced) Bjerrum length to the corresponding real
temperature for `model`'s ion submodel/dielectric constant.
"""
function lB_to_T(model::LS, lB::Real; z=Clapeyron.SA[1.0])
    p = model.ionmodel.params
    ϵr = Clapeyron.dielectric_constant(model.ionmodel.RSPmodel, 1.0, 298.15, z)
    return Clapeyron.e_c^2 / (4π * Clapeyron.ϵ_0 * ϵr * p.sigma * Clapeyron.k_B * lB)
end

"""
    has_physical_loop(model, T, z; nV=400, Vmax_factor=1e5)

Scan pressure over the *physical* volume range (`V > lb_volume`, i.e.
packing fraction `η<1`) at fixed `T` and report whether `P(V)` is
non-monotonic there -- the direct, derivative-free signature of a genuine
two-phase region. Ignores any apparent non-monotonicity below `lb_volume`,
which is just the hard-sphere term diverging at unphysical densities, not a
real phase transition.
"""
function has_physical_loop(model::LS, T::Real, z=Clapeyron.SA[1.0]; nV::Int=400, Vmax_factor::Float64=1e5)
    lbv = Clapeyron.lb_volume(model, T, z)
    Vs = exp.(range(log(lbv * 1.01), log(lbv * Vmax_factor), length=nV))
    prev = Clapeyron.pressure(model, Vs[1], T, z)
    for V in Vs[2:end]
        p = Clapeyron.pressure(model, V, T, z)
        p > prev && return true
        prev = p
    end
    return false
end

"""
    find_critical_point(model; z=SA[1.0])

Find the true critical point of `model` (a single sequence's `LS` model),
via `Clapeyron.crit_pure`. `crit_pure`'s generic initial-guess heuristic
(`x0_crit_pure`, based only on `T_scale`/`lb_volume`) is tuned for
dispersion-driven (van der Waals/SAFT-shaped) transitions with critical
packing fractions around 0.13-0.3; this EOS's phase separation is
electrostatically driven (MSA/Debye-Hückel-like) and has a *much* lower
critical density, so the default guess converges to a spurious stationary
point. Instead: bisect on `T` for the onset of a genuine physical pressure
loop (`has_physical_loop`, itself derivative-free) to get close to the true
critical temperature, then hand `crit_pure` a matching low-packing-fraction
volume guess.

Returns `(Tc, Pc, Vc)`.
"""
function find_critical_point(model::LS; z=Clapeyron.SA[1.0], Tlo_factor::Float64=0.1, Thi_factor::Float64=10.0, max_widen::Int=8)
    Ts = Clapeyron.T_scale(model, z)
    Thi = Ts * Thi_factor  # high T = low lB = expect no coexistence
    Tlo = Ts * Tlo_factor  # low T = high lB = expect coexistence
    has_physical_loop(model, Thi, z) && throw(ArgumentError("no single-phase region found at Thi=$Thi; widen Thi_factor"))
    for _ in 1:max_widen
        has_physical_loop(model, Tlo, z) && break
        Tlo /= 3 # widen toward higher lB (stronger coupling) until coexistence appears
    end
    has_physical_loop(model, Tlo, z) || throw(ArgumentError("no coexistence found even at Tlo=$Tlo; widen Tlo_factor or max_widen"))

    for _ in 1:60
        Tmid = sqrt(Thi * Tlo) # geometric bisection (T spans orders of magnitude)
        if has_physical_loop(model, Tmid, z)
            Tlo = Tmid
        else
            Thi = Tmid
        end
    end
    Tc0 = sqrt(Thi * Tlo)

    # Low critical packing fraction guess, informed by the bisection itself:
    # use lb_volume at the bisected temperature scaled up modestly (electrostatically
    # driven transitions in this EOS have been found ~0.005-0.02 in packing fraction).
    lbv0 = Clapeyron.lb_volume(model, Tc0, z)
    η0 = 0.02
    Vc0 = lbv0 / η0
    x0 = (Tc0 / Ts, log10(Vc0))

    Tc, Pc, Vc = Clapeyron.crit_pure(model, x0, z)
    isnan(Tc) && throw(ErrorException("crit_pure failed to converge from bisection-informed guess"))
    return (Tc=Tc, Pc=Pc, Vc=Vc)
end

"""
    find_coexistence(model, T, crit; z=SA[1.0], guess=nothing)

Locate the two coexisting volumes of `model` at temperature `T`, using
**only** Clapeyron's own machinery — no custom Newton solve at all.

Clapeyron already ships a generic, model-agnostic initial-guess strategy
for exactly this, `Clapeyron.x0_sat_pure_crit(model,T,crit)` (spinodal +
zero-pressure + critical-extrapolation, selected by `T/Tc`, depending only
on `pressure`/`second_virial_coefficient`/`critical_vsat_extrapolation` —
no assumption of a van-der-Waals/SAFT-shaped EOS). It's normally reached
through `x0_sat_pure(model,T,crit)`, but that wrapper has a bug for models
without stored critical-property parameters (it calls the 1-argument
`x0_sat_pure_crit(model)`, which requires a `Tc`/`Pc`/`Vc` `SingleParam`
this model doesn't have, instead of the 3-argument
`x0_sat_pure_crit(model,T,crit)` that actually uses the supplied `crit`) —
so this calls the underlying 3-argument function directly, bypassing the
buggy wrapper. Confirmed to converge directly (no fallback needed) for
~90% of a wide `T` grid; `guess` (the previous temperature step's result,
for continuation) is tried as a fallback for the rest, covering the
remainder.

Returns `(V_dense, V_dilute)`, or `nothing` if `Clapeyron.saturation_pressure`
fails to converge from every guess tried.
"""
function find_coexistence(model::LS, T::Real, crit; z=Clapeyron.SA[1.0], guess=nothing)
    lbv = Clapeyron.lb_volume(model, T, z)
    _, vl0, vv0 = Clapeyron.x0_sat_pure_crit(model, T, crit)
    candidates = filter(!isnothing, ((isnan(vl0) || isnan(vv0)) ? nothing : (vl0, vv0), guess))
    for g in candidates
        p, Vl, Vv = try
            Clapeyron.saturation_pressure(model, T, g)
        catch
            (NaN, NaN, NaN)
        end
        (isnan(p) || Vl <= lbv || Vv <= lbv) && continue # reject unphysical (η>1) solutions too
        # saturation_pressure's (Vl,Vv) aren't guaranteed dense/dilute-ordered
        # for a custom EoS with no inherent liquid/vapor distinction -- sort.
        Vdense, Vdilute = minmax(Vl, Vv)
        return (V_dense=Vdense, V_dilute=Vdilute)
    end
    return nothing
end

"""
    phase_diagram(charges; z=SA[1.0], T_range_factor=0.3, initial_step=0.025, max_iters=2000)

Trace the two-phase coexistence curve for a single sequence, starting from
its true critical point (`find_critical_point`, anchored via
`Clapeyron.crit_pure`) and stepping outward in `T` via **adaptive**
continuation (each step's converged volumes seed the next, exactly like
`find_coexistence`'s own guess chain) down to `T = Tc*T_range_factor`.

A fixed relative step size isn't robust on its own: `Clapeyron.saturation_pressure`
sometimes has a genuinely narrow convergence basin partway through a
sequence's curve (confirmed for the near-alternating sequences, whose much
larger critical Bjerrum length gives a visibly sharper transition) where
neither `x0_sat_pure_crit` nor the previous point's guess converges at the
default step. Rather than reach for a custom free-energy solver to patch
that, the step size here is halved on failure (retrying from the last
converged point) and grown back (capped at `initial_step`) after a
run of successes — plain adaptive continuation, still only ever calling
`Clapeyron.saturation_pressure` for the actual answer.

Returns a `NamedTuple` with `lB`, `rho_dense`, `rho_dilute` (reduced total
bead densities, for comparability with the project's plots) and
`kappa_seq`, plus `Tc`/`lBc` for reference.
"""
function phase_diagram(charges::AbstractVector{<:Integer}; z=Clapeyron.SA[1.0],
                        T_range_factor::Float64=0.3, initial_step::Float64=0.025, max_iters::Int=2000)
    model = LS(charges)
    N = length(charges)
    sigma = LS_SIGMA
    Ts = Clapeyron.T_scale(model, z)

    Tc, Pc, Vc = find_critical_point(model; z=z)
    lBc = Ts / Tc
    crit = (Tc, Pc, Vc)
    Tend = Tc * T_range_factor

    lBs = Float64[]
    ρdense = Float64[]
    ρdilute = Float64[]

    Tcur = Tc
    step = -initial_step * Tc
    guess = nothing
    iters = 0
    while Tcur > Tend && iters < max_iters
        iters += 1
        Tnext = max(Tcur + step, Tend)
        result = find_coexistence(model, Tnext, crit; z=z, guess=guess)
        if result === nothing
            step /= 2
            abs(step) < 1e-10 * Tc && break # stuck; give up on this branch
            continue
        end
        guess = (result.V_dense, result.V_dilute)
        Tcur = Tnext
        step = max(step * 1.5, -initial_step * Tc) # grow back, capped at the initial step
        push!(lBs, Ts / Tcur)
        push!(ρdense, Clapeyron.N_A * z[1] * N * sigma^3 / result.V_dense)
        push!(ρdilute, Clapeyron.N_A * z[1] * N * sigma^3 / result.V_dilute)
    end
    return (lB=lBs, rho_dilute=ρdilute, rho_dense=ρdense, kappa_seq=kappa_seq(charges), Tc=Tc, lBc=lBc)
end

"""
    coexistence_at_lB(charges, lB; z=SA[1.0])

Find the coexistence densities of `charges` at one target Bjerrum length
`lB`, without tracing the full binodal -- for sampling many sequences at a
single fixed `lB` (e.g. testing the disorder-potential hypothesis at fixed
coupling strength) where the full `phase_diagram` continuation would be
unnecessary work. Returns `nothing` if the sequence doesn't phase-separate
at this `lB` (either no critical point could be located within
`find_critical_point`'s search range, or `lB`'s corresponding temperature is
at or above `Tc`, or `find_coexistence` itself fails to converge there) --
otherwise a `NamedTuple` `(lB, rho_dense, rho_dilute, kappa_seq, Tc, lBc)`
matching `phase_diagram`'s per-point fields.
"""
function coexistence_at_lB(charges::AbstractVector{<:Integer}, lB::Real; z=Clapeyron.SA[1.0])
    model = LS(charges)
    N = length(charges)
    sigma = LS_SIGMA
    local Tc, Pc, Vc
    try
        Tc, Pc, Vc = find_critical_point(model; z=z)
    catch
        return nothing
    end
    T = lB_to_T(model, lB; z=z)
    T >= Tc && return nothing
    result = find_coexistence(model, T, (Tc, Pc, Vc); z=z)
    result === nothing && return nothing
    rho_dense = Clapeyron.N_A * z[1] * N * sigma^3 / result.V_dense
    rho_dilute = Clapeyron.N_A * z[1] * N * sigma^3 / result.V_dilute
    lBc = Clapeyron.T_scale(model, z) / Tc
    return (lB=Float64(lB), rho_dense=rho_dense, rho_dilute=rho_dilute, kappa_seq=kappa_seq(charges), Tc=Tc, lBc=lBc)
end
