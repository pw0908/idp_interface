#=
Step 1: single-sequence binodal tracing, via Clapeyron's own (V,T,z)
machinery throughout -- `pressure`, `VT_chemical_potential`, `crit_pure`,
`saturation_pressure` -- so this can never drift from what `a_res` actually
encodes.

`find_critical_point`/`has_physical_loop` (the fixed-ray bisection needed
to seed a good critical-point guess for this EOS's electrostatically-
driven, much-lower-critical-density transitions) now live upstream in
Clapeyron as `LS`-specific `x0_crit_pure`/`x0_crit_mix` methods and the
`has_physical_loop` helper they're built on -- `Clapeyron.crit_pure(model)`
now finds the right answer directly, with no bisection wrapper needed here.
=#

"""
    find_coexistence(model, T, crit; z=SA[1.0], guess=nothing)

Locate the two coexisting volumes of `model` at temperature `T`, via
`Clapeyron.saturation_pressure` (which, now that `LS` has its own
`x0_crit_pure`, builds its own initial guess correctly with no bypass
needed here). `guess` (the previous temperature step's result, for
continuation) is tried as a fallback if the default guess doesn't converge.

Returns `(V_dense, V_dilute)`, or `nothing` if `saturation_pressure` fails
to converge from every guess tried.
"""
function find_coexistence(model::LS, T::Real, crit; z=Clapeyron.SA[1.0], guess=nothing)
    lbv = Clapeyron.lb_volume(model, T, z)
    candidates = guess === nothing ? (nothing,) : (nothing, guess)
    for g in candidates # nothing => saturation_pressure's own default (crit-seeded) guess
        p, Vl, Vv = try
            g === nothing ? Clapeyron.saturation_pressure(model, T; crit=crit) :
                            Clapeyron.saturation_pressure(model, T, g)
        catch
            (NaN, NaN, NaN)
        end
        (isnan(p) || Vl <= lbv || Vv <= lbv) && continue # reject unphysical (η>1) solutions too
        # (Vl,Vv) aren't guaranteed dense/dilute-ordered for a custom EoS
        # with no inherent liquid/vapor distinction -- sort.
        Vdense, Vdilute = minmax(Vl, Vv)
        return (V_dense=Vdense, V_dilute=Vdilute)
    end
    return nothing
end

"""
    phase_diagram(charges; z=SA[1.0], T_range_factor=0.3, initial_step=0.025, max_iters=2000)

Trace the two-phase coexistence curve for a single sequence, starting from
its true critical point (`Clapeyron.crit_pure`) and stepping outward in `T`
via adaptive continuation (each step's converged volumes seed the next)
down to `T = Tc*T_range_factor`.

A fixed relative step size isn't robust on its own: `saturation_pressure`
sometimes has a genuinely narrow convergence basin partway through a
sequence's curve. Rather than reach for a custom free-energy solver to
patch that, the step size here is halved on failure (retrying from the
last converged point) and grown back (capped at `initial_step`) after a
run of successes -- plain adaptive continuation, still only ever calling
`Clapeyron.saturation_pressure` for the actual answer.

Returns a `NamedTuple` with `lB`, `rho_dense`, `rho_dilute` (reduced total
bead densities) and `kappa_seq`, plus `Tc`/`lBc` for reference.
"""
function phase_diagram(charges::AbstractVector{<:Integer}; z=Clapeyron.SA[1.0],
                        T_range_factor::Float64=0.3, initial_step::Float64=0.025, max_iters::Int=2000)
    model = LS(charges)
    N = length(charges)
    sigma = LS_SIGMA
    Ts = Clapeyron.T_scale(model, z)

    Tc, Pc, Vc = Clapeyron.crit_pure(model, nothing, z)
    isnan(Tc) && throw(ErrorException("crit_pure failed to converge for this sequence"))
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
`lB`, without tracing the full binodal. Returns `nothing` if the sequence
doesn't phase-separate at this `lB`.
"""
function coexistence_at_lB(charges::AbstractVector{<:Integer}, lB::Real; z=Clapeyron.SA[1.0])
    model = LS(charges)
    N = length(charges)
    sigma = LS_SIGMA
    local Tc, Pc, Vc
    try
        Tc, Pc, Vc = Clapeyron.crit_pure(model, nothing, z)
        isnan(Tc) && return nothing
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
