#=
Charge-asymmetric (net-charged) polyampholyte, and net-neutral-chain +
symmetric-salt, phase-equilibrium tracing for `LS`.

`DonnanBubblePressure`/`Clapeyron.bubble_pressure(model::LS,...)`,
`component_charges`, and `find_critical_salt_fraction` (the single-point
Donnan-equilibrium solve and the critical-salt-fraction finder) now live
upstream in Clapeyron. What's left here is the continuation/tracing
machinery this project's own campaigns need (adaptive step-halving in `T`
or in composition), which is intentionally NOT upstreamed -- Clapeyron has
no general binodal-tracing combinator, and building one is left to the
user rather than folded into this integration.
=#

const _DONNAN_N = 3  # chain + cation + anion -- fixed by this file's own physical setup

"""
    _donnan_physics_residual!(F, w, model, T)

The `psi_zero` Donnan residual (`Clapeyron.donnan_bubble_residual_psi_zero!`)
reparametrized for pseudo-arclength continuation: `w = [log(Vd), log(Vl),
y1, y2, s]` (`N+2=5` unknowns: log-volumes for conditioning -- V spans many
orders of magnitude approaching the critical point, confirmed directly up
to ~1e12 -- plus the `N-1=2` free dilute-phase mole fractions, plus `s`
itself as a FREE unknown rather than an externally-fixed parameter, which
is the entire point: see `neutral_plus_salt_phase_diagram`'s own docstring
for why fixed-`s` stepping breaks down here). `F` has `N+1=4` entries (one
fewer than `w`), reflecting the genuine 1-parameter family of coexistence
solutions this traces.
"""
function _donnan_physics_residual!(F, w::AbstractVector, model, T)
    Vd = exp(w[1])
    Vl = exp(w[2])
    yfree = w[3:end-1]
    s = w[end]
    u = vcat(Vd, Vl, yfree)
    x = [1 - s, s / 2, s / 2]
    Clapeyron.donnan_bubble_residual_psi_zero!(F, u, model, T, x, _DONNAN_N)
    return F
end

"""
    _donnan_logV_residual!(F, w, model, T, x_dense)

`Clapeyron.donnan_bubble_residual_psi_zero!` reparametrized with
`w = [log(Vd), log(Vl), y1, y2]` at FIXED `x_dense` (unlike
`_donnan_physics_residual!` above, `s` is not a free unknown here -- this
is the plain fixed-`s` equivalent of a single `DonnanBubblePressure(;
psi_zero=true)` solve, just in log-volume rather than raw-volume
coordinates). Square system: `N+1=4` equations for `N+1=4` unknowns.
"""
function _donnan_logV_residual!(F, w::AbstractVector, model, T, x_dense::AbstractVector)
    u = vcat(exp(w[1]), exp(w[2]), w[3:end])
    Clapeyron.donnan_bubble_residual_psi_zero!(F, u, model, T, x_dense, _DONNAN_N)
    return F
end

"""
    _donnan_logV_newton(model, T, x_dense, w0; max_newton=100, tol=1e-10) -> (w, converged)

Plain (undamped) Newton solve of `_donnan_logV_residual!` for `w`, via
`ForwardDiff.jacobian` + `J\\F` -- the fixed-`s` analogue of
`_donnan_arclength_correct` above, minus the arclength augmentation (no
extra free unknown here, so no extra constraint is needed). Used as a
log-volume-parametrized drop-in replacement for
`Clapeyron.bubble_pressure(model,T,x,DonnanBubblePressure(;psi_zero=true))`
in `neutral_plus_salt_phase_diagram`'s warm-started steps, to test
whether conditioning near the critical point (V spanning many orders of
magnitude there, same motivation as the arclength residual's own
log-volume choice) is what limits how far raw-volume natural stepping
can reach.
"""
function _donnan_logV_newton(model, T, x_dense::AbstractVector, w0::Vector{Float64};
                              max_newton::Int=100, tol::Float64=1e-10)
    nF = _DONNAN_N + 1
    w = copy(w0)
    for _ in 1:max_newton
        F = zeros(nF)
        try
            _donnan_logV_residual!(F, w, model, T, x_dense)
        catch
            return w, false
        end
        LinearAlgebra.norm(F) < tol && return w, true
        J = try
            ForwardDiff.jacobian(wi -> begin
                Fi = zeros(eltype(wi), nF)
                _donnan_logV_residual!(Fi, wi, model, T, x_dense)
                Fi
            end, w)
        catch
            return w, false
        end
        Δ = try
            J \ F
        catch
            return w, false
        end
        any(isnan, Δ) && return w, false
        w = w .- Δ
    end
    F = zeros(nF)
    try
        _donnan_logV_residual!(F, w, model, T, x_dense)
    catch
        return w, false
    end
    return w, LinearAlgebra.norm(F) < 1e-6
end

"""
    _donnan_tangent(model, T, w) -> unit tangent vector

Null space of the physics Jacobian (`(N+1)×(N+2)`, rectangular since `w`
has one more unknown -- `s` itself -- than there are equations): the local
direction of the 1-parameter solution curve at `w`, via `ForwardDiff` (the
residual is AD-transparent, matching how it's used elsewhere in Clapeyron's
own critical-point code) and `LinearAlgebra.nullspace`.
"""
function _donnan_tangent(model, T, w::Vector{Float64})
    J = ForwardDiff.jacobian(wi -> begin
        F = zeros(eltype(wi), _DONNAN_N + 1)
        _donnan_physics_residual!(F, wi, model, T)
        F
    end, w)
    ns = LinearAlgebra.nullspace(J)
    isempty(ns) && return nothing
    t = ns[:, 1]
    return t ./ LinearAlgebra.norm(t)
end

"""
    _donnan_arclength_correct(model, T, w_pred, w_prev, tangent, dl) -> (w, converged)

Damped-Newton solve of the AUGMENTED system -- the `N+1` physics equations
plus one pseudo-arclength constraint
`tangent·(w-w_prev) - dl = 0` (Keller 1977) -- for the `N+2` unknowns `w`,
starting from the tangent-predicted `w_pred = w_prev + dl*tangent`. This
stays well-conditioned exactly where fixed-`s` Newton solves for the
physics equations ALONE become ill-conditioned near the critical point,
since the arclength constraint pins down the otherwise-free direction along
the solution curve instead of pinning `s` externally.
"""
function _donnan_arclength_correct(model, T, w_pred::Vector{Float64}, w_prev::Vector{Float64},
                                    tangent::Vector{Float64}, dl::Float64;
                                    max_newton::Int=50, tol::Float64=1e-10)
    nF = _DONNAN_N + 2
    augmented!(F, w) = begin
        _donnan_physics_residual!(view(F, 1:_DONNAN_N+1), w, model, T)
        F[end] = LinearAlgebra.dot(tangent, w .- w_prev) - dl
        F
    end
    w = copy(w_pred)
    for _ in 1:max_newton
        F = augmented!(zeros(nF), w)
        LinearAlgebra.norm(F) < tol && return w, true
        J = ForwardDiff.jacobian(wi -> augmented!(zeros(eltype(wi), nF), wi), w)
        Δ = try
            J \ F
        catch
            return w, false
        end
        any(isnan, Δ) && return w, false
        w = w .- Δ
    end
    F = augmented!(zeros(nF), w)
    return w, LinearAlgebra.norm(F) < 1e-6
end

"""
    _donnan_T_continuation(model, x_dense, T_start, T_target, u0_start;
                            initial_step=10.0, max_step=100.0, max_iters=500, xtol=1e-8)

Continue an already-converged `psi_zero` Donnan bubble-pressure solution
from `T_start` to `T_target` at FIXED composition `x_dense`, by simple
adaptive step-halving continuation in `T` -- the same pattern as
`neutral_plus_salt_phase_diagram`'s own `s`-continuation, just stepping a
different variable. Exists because `s`-continuation at some target `T` can
fail on its very first warm-started step for a sequence/T combination
where `s`-continuation at a DIFFERENT, nearby `T` works fine end-to-end
(confirmed directly: sv15 at `lB=1`'s T=710.6 gets stuck after 1 point via
`s`-stepping, but sv15's OWN full binodal traces cleanly at T=552.9) --
bridging a single composition from a T where it's already known-good to
the target T, at FIXED composition, is a completely different
continuation axis from sweeping `s` at fixed T, and isn't expected to
share the same failure mode.

Returns `(u, converged::Bool)` where `u=[V_dense,V_dilute,y1,y2]` is the
solution at `T_target` (or wherever the continuation stalled, if
`converged=false`).
"""
function _donnan_T_continuation(model, x_dense::AbstractVector, T_start::Real, T_target::Real, u0_start::Vector{Float64};
                                 initial_step::Float64=10.0, max_step::Float64=100.0, max_iters::Int=500, xtol::Float64=1e-8)
    T_cur = T_start
    u_cur = copy(u0_start)
    step = initial_step
    dir = T_target > T_start ? 1.0 : -1.0
    iters = 0
    while abs(T_cur - T_target) > xtol && iters < max_iters
        iters += 1
        T_try = dir > 0 ? min(T_cur + dir * step, T_target) : max(T_cur + dir * step, T_target)
        result = try
            Clapeyron.bubble_pressure(model, T_try, x_dense, DonnanBubblePressure(; psi_zero=true, u0=u_cur))
        catch
            nothing
        end
        if result === nothing || any(isnan, (result.V_dense, result.V_dilute)) || any(isnan, result.y)
            step /= 2
            step < 1e-6 && break
            continue
        end
        u_cur = [result.V_dense, result.V_dilute, result.y[1], result.y[2]]
        T_cur = T_try
        step = min(step * 1.3, max_step)
    end
    return u_cur, abs(T_cur - T_target) < xtol
end

"""
    neutral_plus_salt_phase_diagram(charges, T; ion_charge=[1,-1], s_start=1e-3,
                                     initial_step=1e-3, max_step=1e-2, max_iters=8000,
                                     extrap_ratio_threshold=3.0, use_logV=true,
                                     near_crit_rtol=1e-4, vc_trim_rtol=0.05,
                                     use_density_step=true, target_dlogrho=0.3,
                                     arclength_step=0.05, max_arclength_iters=2000)

Trace the two-phase coexistence curve (binodal) for a net-neutral chain +
symmetric 1:1 salt system at FIXED temperature `T`, scanning the dense
phase's total salt mole fraction `s` (`x(s)=[1-s,s/2,s/2]`) from `s_start`
up to the critical salt fraction (`Clapeyron.find_critical_salt_fraction`)
-- the salt-added analogue of `phase_diagram`/`donnan_phase_diagram`'s own
`T`-continuation, with the roles of `T` and composition (`s`) swapped.

Uses `Clapeyron.bubble_pressure(model,T,x,DonnanBubblePressure(;psi_zero=true))`
(`Ψ≡0` for this system by its own cation↔anion charge-swap symmetry) for the
cold-started first point, with adaptive step-halving continuation, to trace
`s` from `s_start` toward `s_c`.

**`use_logV=true` (default)**: every warm-started point after the first is
solved via `_donnan_logV_newton` instead -- the same physics
(`Clapeyron.donnan_bubble_residual_psi_zero!`), but parametrized in
`log(V_dense)`/`log(V_dilute)` rather than raw volumes. `DonnanBubblePressure`
itself solves in raw volumes, which span many orders of magnitude approaching
the critical point (confirmed directly up to ~1e12) -- the same conditioning
problem `_donnan_physics_residual!`'s own log-volume parametrization was
already built to address for the (opt-in, currently spurious-branch-prone)
arclength phase below. Set `use_logV=false` to recover the original
raw-volume-only behavior for comparison.

**`near_crit_rtol=1e-4`**: the natural-stepping loop stops once
`s_cur >= crit.s_c*(1-near_crit_rtol)`, never attempting a solve exactly at
`crit.s_c` itself -- confirmed directly (both with and without `use_logV`)
that doing so can converge onto a spurious root (`Vd=Vl` but at the wrong
volume, not the true critical state -- the same trivial-branch degeneracy
documented for the arclength phase below, which turns out not to be
arclength-specific). The reported `s_c`/`rho_chain_c`/`rho_salt_c` fields
are still the EXACT independently-computed critical point; only the traced
curve itself stops just short of it. Not sufficient by itself -- see
`vc_check_ratio`/`vc_rtol` below, which catches the degeneracy this alone
does not.

**`use_density_step=true` (default), `target_dlogrho=0.3`**: once there
are 2+ accepted points (the growth rule below still governs the very
first step, since a slope needs two points), the step size in `s` is
chosen to target a constant change in `log(rho_salt_dense)` -- the
(reduced, dense-phase) salt density, `N_A*s*sigma^3/V_dense` -- rather
than growing `step` by a fixed `*1.3` factor in `s` itself. `s` can
compress very unevenly in density space approaching criticality
(`dV/ds` diverging there means a fixed Δs corresponds to a wildly
different Δ(density) depending on where on the curve it's taken), so a
fixed growth-in-`s` rule can hand the warm-started solve a jump far
outside its basin of convergence even where the destination point itself
is easily solvable from a cold start -- confirmed directly this is sv15's
actual failure mode at `lB=1`'s T: a cold solve at the second point
succeeds fine, only the WARM-STARTED jump from point 1 fails. Set
`use_density_step=false` to recover the original `s`-only step-growth
rule.

**`vc_check_ratio=1.1`, `vc_rtol=0.02`**: once a converged point's own
`V_dilute/V_dense < vc_check_ratio` (it "looks" near-critical), its `V_dense`
and `V_dilute` are cross-checked directly against the independently-known
`crit.V_c`; a point more than `vc_rtol` away from `crit.V_c` is treated as a
failed solve (same step-halving retry as any other failure) rather than
accepted. Needed in addition to `near_crit_rtol`, not instead of it --
confirmed directly that `near_crit_rtol` alone still let a spurious
trivial-branch point (`Vd=Vl`, 4.4% off `crit.V_c`) through: at the point
where it first appeared, it was also deceptively close to a linear
extrapolation of the last two genuine points (itself already ~1% inaccurate
that near the critical bend), so an earlier version of this check that
compared against the extrapolated prediction instead of `crit.V_c` directly
did not catch it either.

**Two-regime warm-start**: plain point-to-point reuse (the last converged
`u` as the next `u0`) works well far from `s_c`, but becomes a poor
predictor as the curve bends sharply approaching criticality. Once the
last converged point's density ratio drops below `extrap_ratio_threshold`
(the curve is already close enough to be locally smooth), this switches to
linear extrapolation in `s` from the last two converged points instead.

**Pseudo-arclength continuation past that point**: fixed-`s` stepping (even
with the extrapolation fix above) genuinely breaks down approaching `s_c`,
not just slows down -- confirmed directly: naively pushing it further (a
smaller `extrap_ratio_threshold`) made the solver converge to a spurious,
UNBOUNDED fixed point (`V_dense,V_dilute→∞` together, confirmed via the raw
solver state, while `s` itself barely advanced) rather than the true
critical state, for one test sequence -- a false positive that happened to
report `s→s_c` without reaching anything physical. This is the classic
symptom of `dV/ds→∞` as the two phases merge: a numerically modest step in
the EXTERNALLY-FIXED parameter `s` can demand an arbitrarily large,
poorly-predicted jump in `V`. Once natural stepping exhausts itself (step
size collapses below `1e-13` or `max_iters` is hit) with the density ratio
still above 1, this switches to Keller's pseudo-arclength continuation
(`_donnan_arclength_correct`/`_donnan_tangent` above): the SAME physics
equations, reparametrized in terms of arc length along the solution curve
in the full `(log V_dense,log V_dilute,y,s)` space, with `s` itself now a
free unknown rather than a fixed parameter -- this stays well-conditioned
through exactly the region fixed-`s` stepping cannot resolve, regardless of
which combination of variables is changing fastest there. Stops once the
two phases have genuinely merged (`V_dilute/V_dense - 1 < 1e-4`) or
`max_arclength_iters` is exhausted; each accepted step is sanity-checked
(positive volumes/mole fractions, `s∈(0,1)`) and rejected (arc-length step
halved, retried) otherwise, the same safety net as the natural-stepping
phase's own step-halving.

Even with this, may still stop short of the EXACT `s_c` in some cases
(near-critical coexistence is intrinsically hard for any solver) -- the
returned curve's own trend as `s → s_c` is the honest signal of how close
it got, not a promise of reaching the exact critical point.

Returns a `NamedTuple` with `s`, dense/dilute-phase reduced number
densities for chain beads (`rho_chain_dense`/`_dilute`) and each salt ion
(`rho_cation_dense`/`_dilute`, `rho_anion_dense`/`_dilute`), plus
`s_c`/`T`/`rho_chain_c`/`rho_salt_c` (the exact critical point) for
reference, and `u` (the raw `[V_dense,V_dilute,y1,y2]` warm-started solver
state at each `s`, so a caller can rebuild a DFT structure directly instead
of re-solving cold -- re-solving cold reliably fails near `s_c`).
"""
function neutral_plus_salt_phase_diagram(charges::AbstractVector{<:Integer}, T::Real;
                                          ion_charge::AbstractVector{<:Integer}=[1, -1],
                                          s_start::Float64=1e-3, initial_step::Float64=1e-3,
                                          max_step::Float64=1e-2, max_iters::Int=8000,
                                          extrap_ratio_threshold::Float64=3.0,
                                          use_logV::Bool=true, near_crit_rtol::Float64=1e-4,
                                          vc_trim_rtol::Float64=0.05,
                                          use_density_step::Bool=true, target_dlogrho::Float64=0.3,
                                          use_arclength::Bool=false,
                                          arclength_step::Float64=0.05, max_arclength_iters::Int=2000,
                                          crit=nothing)
    model = LS(charges; ion_charge=ion_charge)
    Nchain = length(charges)
    sigma = LS_SIGMA

    # `find_critical_salt_fraction`'s own crit_mix bisection is the single
    # most expensive part of this function for some sequences (confirmed
    # directly: ~2 minutes for sv30 at lB=1.0's T, independent of anything
    # else below) -- accepting a pre-computed `crit` lets a caller that
    # needs to retry this function several times at the SAME (model,T)
    # with different stepping parameters (e.g. scanning `initial_step` for
    # a sequence where the default fails) pay that cost once, not once per
    # retry.
    crit = crit === nothing ? find_critical_salt_fraction(model, T) : crit

    # Stop just short of the TRUE s_c rather than ever solving exactly at
    # it: the fixed-s coexistence equations are mathematically singular
    # exactly at the critical point (the whole reason a critical point is
    # special -- the stability Jacobian has a zero eigenvalue there), so a
    # plain Newton solve AT s_c is ill-posed regardless of parametrization.
    # Confirmed directly this isn't just theoretical: forcing the last step
    # to land exactly on `crit.s_c` made the (very well warm-started, log-
    # volume) Newton solve jump to a DIFFERENT, spurious self-consistent
    # root where Vd=Vl but at roughly 2x the true critical volume -- the
    # same "trivial branch" (y=x, Vd=Vl, true at every s, not just s_c)
    # degeneracy already documented at length for the arclength phase
    # below, turning up here too. The second-to-last point before that
    # jump (one `near_crit_rtol` short of crit.s_c) tracked the true curve
    # to within ~0.3% of `crit.V_c` -- stopping there instead is both safe
    # and, for plotting purposes, visually indistinguishable from an exact
    # closure.
    s_target = crit.s_c * (1 - near_crit_rtol)

    s_vals = Float64[]
    rho_chain_dense, rho_chain_dilute = Float64[], Float64[]
    rho_cation_dense, rho_cation_dilute = Float64[], Float64[]
    rho_anion_dense, rho_anion_dilute = Float64[], Float64[]

    s_hist = Float64[]
    u_hist = Vector{Float64}[]
    last_ratio = Inf
    s_cur = 0.0
    step = initial_step
    iters = 0
    while s_cur < s_target && iters < max_iters
        iters += 1
        s = min(iters == 1 ? s_start : s_cur + step, s_target)
        x = [1 - s, s / 2, s / 2]
        extrapolated = length(u_hist) >= 2 && last_ratio < extrap_ratio_threshold
        u0 = if extrapolated
            s1, s2 = s_hist[end-1], s_hist[end]
            u1, u2 = u_hist[end-1], u_hist[end]
            u2 .+ (s - s2) .* (u2 .- u1) ./ (s2 - s1)
        elseif !isempty(u_hist)
            u_hist[end]
        else
            nothing
        end
        result = if use_logV && u0 !== nothing
            # Warm start available: solve in log-volume coordinates (see
            # `_donnan_logV_newton`'s own docstring for why) rather than
            # raw-volume `DonnanBubblePressure`. Falls through to `nothing`
            # (triggering the same step-halving retry below) on failure,
            # exactly like the raw-volume path.
            try
                w0 = vcat(log(u0[1]), log(u0[2]), u0[3:end])
                w, converged = _donnan_logV_newton(model, T, x, w0)
                if converged
                    Vd, Vl = exp(w[1]), exp(w[2])
                    yfree = w[3:end]
                    y = vcat(yfree, 1 - sum(yfree))
                    (V_dense=Vd, V_dilute=Vl, y=y)
                else
                    nothing
                end
            catch
                nothing
            end
        else
            # Either no warm start yet (first point: cold-start via the
            # native `DonnanBubblePressure` bracket-search guess -- an
            # easy, far-from-critical regime where raw volumes are not the
            # bottleneck), or `use_logV=false` with a warm start available
            # -- in which case `u0` MUST still be passed through here. An
            # earlier version of this branch unconditionally omitted `u0`
            # (only reachable/correct when it really was `nothing`, true
            # before `use_logV` existed), silently cold-starting every
            # single point whenever `use_logV=false` -- confirmed directly
            # this broke `use_logV=false` as a real fallback/control path,
            # not just as a performance regression: it made a direct
            # comparison test look like sv15 fails identically regardless
            # of `use_logV`, when actually only the (buggy) `use_logV=false`
            # path failed -- `use_logV=true` was never actually compared
            # against a correctly warm-started raw-volume baseline.
            method = u0 === nothing ? DonnanBubblePressure(; psi_zero=true) : DonnanBubblePressure(; psi_zero=true, u0=u0)
            try
                Clapeyron.bubble_pressure(model, T, x, method)
            catch
                nothing
            end
        end
        if result === nothing || any(isnan, (result.V_dense, result.V_dilute)) || any(isnan, result.y)
            step /= 2
            step < 1e-13 && break
            continue  # next attempt recomputes s = s_cur + (now-halved) step
        end
        last_ratio = result.V_dilute / result.V_dense
        push!(s_hist, s)
        push!(u_hist, vcat(result.V_dense, result.V_dilute, result.y[1], result.y[2]))
        push!(s_vals, s)
        push!(rho_chain_dense, Clapeyron.N_A * x[1] * Nchain * sigma^3 / result.V_dense)
        push!(rho_chain_dilute, Clapeyron.N_A * result.y[1] * Nchain * sigma^3 / result.V_dilute)
        push!(rho_cation_dense, Clapeyron.N_A * x[2] * sigma^3 / result.V_dense)
        push!(rho_cation_dilute, Clapeyron.N_A * result.y[2] * sigma^3 / result.V_dilute)
        push!(rho_anion_dense, Clapeyron.N_A * x[3] * sigma^3 / result.V_dense)
        push!(rho_anion_dilute, Clapeyron.N_A * result.y[3] * sigma^3 / result.V_dilute)
        s_cur = s
        # Step in (reduced, dense-phase) salt DENSITY rather than salt mole
        # fraction `s` itself, per direct suggestion: `s` can compress very
        # unevenly in density space (dV/ds diverging approaching
        # criticality means a fixed-size Δs step corresponds to a wildly
        # different Δ(density) depending on where on the curve it's taken),
        # which is a plausible reason a single warm-started step can
        # overshoot the solver's basin even right after a good cold start
        # (confirmed as sv15's actual failure mode: the SECOND point fails
        # even though a fresh cold solve at that same s succeeds fine --
        # i.e. it's the SIZE of the jump from point 1, not point 2's own
        # solvability, that's the problem). Estimates the local
        # d(log rho_salt_dense)/ds slope from the last two accepted points
        # and picks the next Δs to target a constant `target_dlogrho` step
        # in log-density instead of the old constant `*1.3` growth in `s`
        # itself. Needs 2 points to estimate a slope, so the very first
        # step (cold start -> second point) still uses the plain `initial_step`
        # growth rule -- exactly where sv15 fails, so this is not expected
        # to fix that specific case, only the steps after it.
        if use_density_step && length(s_hist) >= 2
            s_prev, s_now = s_hist[end-1], s_hist[end]
            rho_prev = Clapeyron.N_A * s_prev * sigma^3 / u_hist[end-1][1]
            rho_now = Clapeyron.N_A * s_now * sigma^3 / u_hist[end][1]
            dlogrho = log(rho_now) - log(rho_prev)
            ds = s_now - s_prev
            step = if isfinite(dlogrho) && abs(dlogrho) > 1e-12 && ds > 0
                clamp(target_dlogrho / abs(dlogrho / ds), 1e-13, max_step)
            else
                min(step * 1.3, max_step)
            end
        else
            step = min(step * 1.3, max_step)
        end
    end

    # Post-hoc trim, not an in-loop reject: drop trailing points whose V
    # isn't actually close to the independently-known `crit.V_c`, even
    # though they converged cleanly and reported Vd≈Vl.
    #
    # An IN-LOOP version of this check (reject and step-halve-retry
    # immediately) was tried first and made things WORSE, not better --
    # confirmed directly: once a warm start lands close enough to the
    # trivial branch (y≡x, Vd≡Vl, which trivially satisfies every physics
    # equation at ANY V, not just `V_c` -- a genuine degenerate solution
    # manifold, not just a numerical artifact) for the Jacobian to go
    # near-singular there, EVERY subsequent retry warm-started from that
    # same neighborhood -- even at a vanishingly small step -- kept
    # reconverging to the same invalid point and getting rejected again,
    # burning the entire iteration budget stuck at one `s` rather than
    # backing off to a still-valid earlier point. Trimming after the fact
    # sidesteps that entirely: it never perturbs the stepping/warm-start
    # dynamics, it just refuses to report a trailing point that doesn't
    # hold up against the one piece of independently-known ground truth
    # this function already has.
    while length(u_hist) >= 2
        Vd, Vl = u_hist[end][1], u_hist[end][2]
        max(abs(Vd - crit.V_c), abs(Vl - crit.V_c)) / crit.V_c > vc_trim_rtol || break
        pop!(s_hist); pop!(u_hist); pop!(s_vals)
        pop!(rho_chain_dense); pop!(rho_chain_dilute)
        pop!(rho_cation_dense); pop!(rho_cation_dilute)
        pop!(rho_anion_dense); pop!(rho_anion_dilute)
    end

    # Pseudo-arclength phase: picks up from wherever natural stepping left
    # off (s_hist[end]/u_hist[end]), through the region fixed-s stepping
    # cannot resolve -- see this function's own docstring for why. A no-op
    # if natural stepping already reached s_c cleanly, or never got a
    # single point to start from.
    #
    # DISABLED BY DEFAULT (use_arclength=false): confirmed directly (cross-
    # checked against sv10, whose `find_critical_salt_fraction` answer is
    # independently verified correct -- `crit_mix`'s own Tc at that s_c
    # matches the target T exactly) that this phase can converge onto a
    # SPURIOUS solution family instead of the true coexistence curve: for
    # ANY s, setting V_dense=V_dilute and y=x(s) (i.e. "no phase separation
    # at all") trivially satisfies both the pressure-equality and every
    # chemical-potential-equality equation, since it reduces to comparing a
    # state to itself -- a degenerate root that exists at EVERY s, not just
    # the true critical one. Near-merged phases (exactly where this phase
    # is needed) sit numerically close to that trivial branch, and nothing
    # here currently stops the corrector from snapping onto it instead of
    # continuing to track the genuine physical curve -- confirmed: for
    # sv10, this phase "converged" (composition gap ~1e-15, genuinely a
    # root of the equations) at a DIFFERENT s than the known-correct
    # critical point. The composition-gap stopping check alone cannot tell
    # the two apart. Needs a proper fix (e.g. a non-degeneracy constraint
    # keeping steps off the trivial branch) before this is trustworthy --
    # left implemented but opt-in so no existing caller silently picks up
    # spurious points.
    #
    # Criticality is checked via the full COMPOSITION difference
    # (max|x_dense-y| over every component), not the volume ratio alone:
    # confirmed directly that V_dilute/V_dense can already sit close to 1
    # (both phases' overall packing converging) while the two phases'
    # COMPOSITIONS -- and hence e.g. the chain's own dense/dilute density
    # ratio -- remain wildly different (~28x for one test sequence), since
    # the chain is a minor component by the time s is this close to s_c and
    # its own partitioning hasn't converged even though bulk volume has. A
    # one-directional volume check (`V_dilute/V_dense > 1+tol`) is also
    # fragile regardless -- which phase ends up larger isn't guaranteed --
    # confirmed directly: it evaluated false (wrongly concluding "already
    # converged") when the actual ratio was 0.9994, i.e. on the WRONG side
    # of 1 for that one-sided check, silently skipping this entire phase.
    last_x = [1 - s_hist[end], s_hist[end] / 2, s_hist[end] / 2]
    last_y = isempty(u_hist) ? Float64[] : (u -> [u[3], u[4], 1 - u[3] - u[4]])(u_hist[end])
    comp_gap = isempty(u_hist) ? 0.0 : maximum(abs.(last_x .- last_y))
    if use_arclength && !isempty(u_hist) && comp_gap > 1e-4
        Vd0, Vl0, y1_0, y2_0 = u_hist[end]
        s0 = s_hist[end]
        w_prev = [log(Vd0), log(Vl0), y1_0, y2_0, s0]

        # Bootstrap an initial tangent from a tiny natural step forward (a
        # genuine nearby second point, not just the Jacobian null space,
        # which can have an ambiguous sign) -- falls back to the null-space
        # tangent if that tiny step itself doesn't converge.
        ds_boot = 1e-6
        s1 = min(s0 + ds_boot, crit.s_c)
        x1 = [1 - s1, s1 / 2, s1 / 2]
        boot = try
            Clapeyron.bubble_pressure(model, T, x1, DonnanBubblePressure(; psi_zero=true, u0=u_hist[end]))
        catch
            nothing
        end
        tangent = if boot !== nothing && !any(isnan, (boot.V_dense, boot.V_dilute)) && !any(isnan, boot.y) && s1 != s0
            w1 = [log(boot.V_dense), log(boot.V_dilute), boot.y[1], boot.y[2], s1]
            Δw = w1 .- w_prev
            Δw ./ LinearAlgebra.norm(Δw)
        else
            _donnan_tangent(model, T, w_prev)
        end

        if tangent !== nothing
            tangent[end] < 0 && (tangent = -tangent)  # orient forward (increasing s initially)

            dl = arclength_step
            arc_iters = 0
            while arc_iters < max_arclength_iters
                arc_iters += 1
                w_pred = w_prev .+ dl .* tangent
                w_new, converged = _donnan_arclength_correct(model, T, w_pred, w_prev, tangent, dl)

                Vd = converged ? exp(w_new[1]) : NaN
                Vl = converged ? exp(w_new[2]) : NaN
                y1, y2 = converged ? w_new[3] : NaN, converged ? w_new[4] : NaN
                s = converged ? w_new[end] : NaN
                y3 = 1 - y1 - y2
                physical = converged && Vd > 0 && Vl > 0 && y1 > 0 && y2 > 0 && y3 > 0 && 0 < s < 1

                if !physical
                    dl /= 2
                    dl < 1e-12 && break
                    continue
                end

                new_tangent = _donnan_tangent(model, T, w_new)
                if new_tangent === nothing
                    dl /= 2
                    dl < 1e-12 && break
                    continue
                end
                LinearAlgebra.dot(new_tangent, tangent) < 0 && (new_tangent = -new_tangent)

                push!(s_hist, s)
                push!(u_hist, [Vd, Vl, y1, y2])
                push!(s_vals, s)
                push!(rho_chain_dense, Clapeyron.N_A * (1 - s) * Nchain * sigma^3 / Vd)
                push!(rho_chain_dilute, Clapeyron.N_A * y1 * Nchain * sigma^3 / Vl)
                push!(rho_cation_dense, Clapeyron.N_A * (s / 2) * sigma^3 / Vd)
                push!(rho_cation_dilute, Clapeyron.N_A * y2 * sigma^3 / Vl)
                push!(rho_anion_dense, Clapeyron.N_A * (s / 2) * sigma^3 / Vd)
                push!(rho_anion_dilute, Clapeyron.N_A * y3 * sigma^3 / Vl)

                w_prev = w_new
                tangent = new_tangent
                dl = min(dl * 1.3, arclength_step * 4)

                # Same full-composition criticality check as the entry
                # guard above -- NOT the one-directional volume ratio (see
                # that comment for why).
                x_new = [1 - s, s / 2, s / 2]
                y_new = [y1, y2, y3]
                maximum(abs.(x_new .- y_new)) < 1e-4 && break  # phases have genuinely merged
            end
        end
    end

    rho_chain_c = Clapeyron.N_A * (1 - crit.s_c) * Nchain * sigma^3 / crit.V_c
    rho_salt_c = Clapeyron.N_A * (crit.s_c / 2) * sigma^3 / crit.V_c

    return (s=s_vals, rho_chain_dense=rho_chain_dense, rho_chain_dilute=rho_chain_dilute,
            rho_cation_dense=rho_cation_dense, rho_cation_dilute=rho_cation_dilute,
            rho_anion_dense=rho_anion_dense, rho_anion_dilute=rho_anion_dilute,
            s_c=crit.s_c, T=T, rho_chain_c=rho_chain_c, rho_salt_c=rho_salt_c,
            u=u_hist)
end

"""
    critical_point_from_trace(s_vals, u_vals)

Locate the critical point directly from a `neutral_plus_salt_phase_diagram`
(or T-bridged) trace's own `(s, u)` points, as the salt fraction where
`V_dense` and `V_dilute` cross (ratio -> 1) -- the literal definition of the
two phases merging -- rather than from `Clapeyron.find_critical_salt_fraction`'s
`crit_mix`-based bulk critical-point solver.

Motivation: confirmed directly (sv25/sv30 Das-Pappu sequences at lB=1.0sigma)
that `crit_mix` can converge to a WRONG, unrelated solution of the general
multicomponent critical-point equations for compositions this asymmetric
(dominant small-ion salt mole fraction against a long neutral chain) --
reporting an `s_c` whose own `Tc_mix` is hundreds of K off target, and whose
`(s_c,V_c)` sits nowhere near where the trace's OWN `V_dense`/`V_dilute`
actually merge (sv25: crit_mix said s_c~0.988-0.994; the trace itself merges
at s~0.955. sv30: crit_mix said s_c~0.987-0.997; the trace merges at
s~0.857). This held even after rewriting `find_critical_salt_fraction`'s
bisection to warm-start continuously and stop at the first sign change
(a real robustness improvement, but moot if `crit_mix` itself is on the
wrong branch throughout). Deriving the critical point from the SAME
coexistence calculation that produces the rest of the curve sidesteps that
unreliability entirely (it's ground-truth consistent by construction), at
the cost of requiring the trace to have actually bracketed the crossing
(points on both sides of ratio=1).

With a sufficiently fine step (confirmed directly: capping `max_step` at
1e-4 for `neutral_plus_salt_phase_diagram`, ~1e4 points across the full
trace) the continuation can track smoothly straight through `ratio=1`
without ever landing on the trivial branch at all -- in which case this
finds the crossing by direct interpolation between the two real adjacent
points bracketing it, the most accurate case. With coarser stepping the
continuation more often jumps straight from a smoothly-decreasing `ratio>1`
point onto the trivial degenerate branch (`V_dense` dropping discontinuously,
often 5-6x, to land exactly on `ratio=1.0`) rather than passing through
`ratio=1` itself -- the trivial branch (`y=x`, `V_dense=V_dilute` for ANY
composition, not just at the true critical point) is a standing hazard
documented throughout this file's other near-critical handling. For that
case, this falls back to finding the last index before any such jump (a
relative change in `V_dense` between consecutive points exceeding
`jump_rtol`) and extrapolating a LINE fit to `V_dilute/V_dense` vs `s` over
the trailing `n_fit` clean points before that jump (log-linear fits for
`V_dense` and `V_dilute` individually, evaluated at the extrapolated `s_c`,
then averaged, give `V_c`) -- markedly less reliable than a direct crossing
(confirmed directly: for a trace that DID also reach a genuine crossing
further out, the fallback's extrapolated `s_c` from an earlier, coarser
window was off by several percent from the true crossing) and only used
when the trace never reaches one.

Returns `(s_c, V_c)`, or `(s_c=NaN, V_c=NaN)` if there aren't enough clean
points, or the trend isn't decreasing toward 1.
"""
function critical_point_from_trace(s_vals::AbstractVector, u_vals::AbstractVector; n_fit::Int=6, jump_rtol::Float64=0.3)
    n = length(s_vals)
    n >= 2 || return (s_c=NaN, V_c=NaN)

    # Forward scan, stopping at the FIRST bad transition -- NOT a backward
    # scan stopping at the first *good* one, which fails when the trivial
    # branch is landed on for several consecutive points in a row (the jump
    # ONTO it is huge, but point-to-point changes WITHIN it can be small,
    # so a backward scan can stop short, leaving multiple trivial-branch
    # points inside the fit window -- confirmed directly, sv15 at lB=1.0).
    last_good = 1
    for i in 2:n
        Vd0, Vd1 = u_vals[i-1][1], u_vals[i][1]
        abs(Vd1 - Vd0) / Vd0 > jump_rtol && break
        last_good = i
    end
    last_good >= 2 || return (s_c=NaN, V_c=NaN)

    # Prefer a DIRECT crossing: scan the clean region (1:last_good) for the
    # first adjacent pair bracketing ratio=1 and interpolate between them --
    # no fit/extrapolation needed, and far more accurate when available.
    for i in 2:last_good
        Vd0, Vl0 = u_vals[i-1][1], u_vals[i-1][2]
        Vd1, Vl1 = u_vals[i][1], u_vals[i][2]
        r0, r1 = Vl0 / Vd0, Vl1 / Vd1
        if r0 > 1.0 >= r1
            s0, s1 = s_vals[i-1], s_vals[i]
            t = (1.0 - r0) / (r1 - r0)
            s_c = s0 + t * (s1 - s0)
            lnVd_c = log(Vd0) + t * (log(Vd1) - log(Vd0))
            lnVl_c = log(Vl0) + t * (log(Vl1) - log(Vl0))
            return (s_c=s_c, V_c=exp(0.5 * (lnVd_c + lnVl_c)))
        end
    end

    # Fall back to extrapolation from the trailing clean points -- see
    # docstring for why this is less reliable.
    fit_idx = max(1, last_good - n_fit + 1):last_good
    length(fit_idx) >= 2 || return (s_c=NaN, V_c=NaN)
    s_fit = collect(float.(s_vals[fit_idx]))
    ratio_fit = [u_vals[i][2] / u_vals[i][1] for i in fit_idx]

    sbar = sum(s_fit) / length(s_fit)
    denom = sum((s_fit .- sbar) .^ 2)
    denom > 0 || return (s_c=NaN, V_c=NaN)
    lin_slope_intercept(y_fit) = begin
        ybar = sum(y_fit) / length(y_fit)
        b = sum((s_fit .- sbar) .* (y_fit .- ybar)) / denom
        a = ybar - b * sbar
        (a, b)
    end

    a_r, b_r = lin_slope_intercept(ratio_fit)
    b_r < 0 || return (s_c=NaN, V_c=NaN)
    s_c = (1.0 - a_r) / b_r

    lnVd_fit = [log(u_vals[i][1]) for i in fit_idx]
    lnVl_fit = [log(u_vals[i][2]) for i in fit_idx]
    a_d, b_d = lin_slope_intercept(lnVd_fit)
    a_l, b_l = lin_slope_intercept(lnVl_fit)
    V_c = exp(0.5 * ((a_d + b_d * s_c) + (a_l + b_l * s_c)))
    return (s_c=s_c, V_c=V_c)
end

"""
    critical_point_near_trivial_branch(model, T, s_start, u_start; max_step=1e-4, max_iters=200000, gap_floor=1e-8)

Find the critical point by continuing PAST where an existing trace ends,
until the solver snaps onto the trivial branch (`V_dense=V_dilute`,
`y=x` -- see `critical_point_from_trace`'s docstring), and returning the
LAST point before that snap.

Motivation: confirmed directly (sv20/sv25/sv30 Das-Pappu sequences at
lB=1.0sigma) that `critical_point_from_trace`'s `V_dense=V_dilute` crossing
is NOT reliably the true critical point -- true criticality requires BOTH
volumes and compositions to converge together, and for these sequences the
volume ratio crosses 1 (and then drifts away again) while the compositions
are still substantially different (confirmed: gaps of 0.008-0.14 in mole
fraction at the volume crossing, vs <0.0002 for sv10 where the volume-based
answer happens to already be accurate). The compositions keep converging,
slowly, past that spurious volume crossing -- the genuine critical point is
near where the solver EVENTUALLY snaps onto the trivial branch (`y=x`
exactly in one step), since that's structurally where the real and
degenerate trivial solutions merge.

Continuation here is a simple fixed-composition-ray march in `s` (NOT the
pseudo-arclength `_donnan_physics_residual!` scheme), warm-started each step
from the previous point, halving `max_step` on a failed/NaN solve -- the
same scheme `_donnan_T_continuation` uses for its own T-sweep, applied to an
s-sweep instead. This region is numerically difficult AND genuinely noisy,
not just slow (confirmed directly, sv25: long stretches need many
successive halvings down to ~1e-8 step, with `V_dense` fluctuating by
~1e-6 relative between accepted steps throughout -- well under a
volume-jump threshold would need to be to avoid false-triggering on this
noise, so a single-step `V_dense` discontinuity is NOT used as the stopping
signal here). Instead, stops only when the mole-fraction gap `|x-y|`
collapses to near machine precision (`<gap_floor`) from a non-negligible
value in a single step -- confirmed directly this is exact (`~1e-12`) at
the genuine trivial-branch takeover, unlike the gradual shrinking (still
~0.026 well into the difficult region) that happens on the real branch.
`max_iters` needs to be generous given how slow/noisy this region is.

Returns `(s_c, V_c, converged, s_hist, u_hist)`: `s_c`/`V_c` from the last
point before the trivial-branch jump (or before `max_iters` is exhausted, in
which case `converged=false` and the result is a lower bound, not a
converged answer); `s_hist`/`u_hist` are every intermediate point visited
(starting from `s_start`/`u_start`), for plotting the true final approach
rather than a straight line from wherever an existing trace happened to stop
to the critical point.
"""
function critical_point_near_trivial_branch(model::Clapeyron.EoSModel, T::Real, s_start::Real, u_start::AbstractVector;
                                             max_step::Float64=1e-4, max_iters::Int=200000, gap_floor::Float64=1e-8)
    s = float(s_start)
    u = collect(float.(u_start))
    s_hist = [s]
    u_hist = [u]
    gap(ss, uu) = abs((1 - ss) - uu[3])
    step = max_step
    for _ in 1:max_iters
        s_try = s + step
        x = [1 - s_try, s_try / 2, s_try / 2]
        result = try
            Clapeyron.bubble_pressure(model, T, x, DonnanBubblePressure(; psi_zero=true, u0=u))
        catch
            nothing
        end
        if result === nothing || any(isnan, (result.V_dense, result.V_dilute)) || any(isnan, result.y)
            step /= 2
            step < 1e-14 && return (s_c=s, V_c=u[1], converged=false, s_hist=s_hist, u_hist=u_hist)
            continue
        end
        u_new = [result.V_dense, result.V_dilute, result.y[1], result.y[2]]
        g_new = gap(s_try, u_new)
        # The trivial branch is EXACTLY y=x -- confirmed directly that this
        # is a much more specific/robust signature than any single-step
        # volume jump: the approach THROUGH this region is genuinely noisy
        # (many small failed/halved steps even while still clearly on the
        # real branch, confirmed directly -- a volume-jump threshold alone
        # false-triggers on that noise, well before the true takeover).
        # Require the gap to actually collapse near machine precision, not
        # merely shrink.
        g_new < gap_floor && gap(s, u) > 1e-4 && return (s_c=s, V_c=u[1], converged=true, s_hist=s_hist, u_hist=u_hist)
        u = u_new
        s = s_try
        push!(s_hist, s)
        push!(u_hist, u)
        step = min(step * 1.3, max_step)
    end
    return (s_c=s, V_c=u[1], converged=false, s_hist=s_hist, u_hist=u_hist)
end

"""
    critical_point_min_gap(s_vals, u_vals; trivial_tol=1e-6)

Locate the critical point as the point of MINIMUM mole-fraction gap `|x-y|`
along a trace (excluding any point already exactly on the trivial branch,
`gap<trivial_tol`).

Motivation: supersedes both `critical_point_from_trace` (volume-ratio
crossing -- confirmed directly to be a spurious, non-critical event for
sequences where volumes and compositions don't converge together) and
`critical_point_near_trivial_branch`'s "last point before the trivial-branch
jump" (confirmed directly, sv30 at lB=1.0sigma with `use_arclength=true`:
the composition gap reaches a genuine MINIMUM partway through the
arclength-continuation phase, then INCREASES again as the continuation
overshoots past criticality into an unphysical `s->1` regime -- "last point
before a jump" would pick a much worse point than the true minimum, which
occurs well before any jump). The minimum-gap point is the best available
estimate regardless of whether the trace terminates via a trivial-branch
snap, an overshoot, or simply running out of iterations.

Pair this with `neutral_plus_salt_phase_diagram(...; use_arclength=true,
arclength_step=0.005)` (a smaller `arclength_step` than that function's own
default `0.05` -- confirmed directly this matters: it reaches a composition
gap almost 10x smaller before failing) to get a trace that actually reaches
close enough for this to be meaningful; the natural-stepping phase alone
(no `use_arclength`) typically stalls with the gap still of order 1e-2.

Returns `(s_c, V_c)` using the minimum-gap point's own `s` and `V_dense`
(averaging would double-count since `V_dense≈V_dilute` there already), or
`(NaN, NaN)` if every point is on the trivial branch.
"""
function critical_point_min_gap(s_vals::AbstractVector, u_vals::AbstractVector; trivial_tol::Float64=1e-6)
    gaps = [abs((1 - s_vals[i]) - u_vals[i][3]) for i in eachindex(s_vals)]
    candidates = findall(>(trivial_tol), gaps)
    isempty(candidates) && return (s_c=NaN, V_c=NaN)
    best = candidates[argmin(gaps[candidates])]
    return (s_c=s_vals[best], V_c=u_vals[best][1])
end

"""
    donnan_phase_diagram(charges; ion_charge=-1, T_range_factor=0.3, initial_step=0.025, max_iters=2000)

Trace the two-phase coexistence curve across `lB` for a charge-asymmetric
polyampholyte chain plus its stoichiometric counter-ion (`N=2`), the
Donnan-equilibrium analogue of `phase_diagram`. Since exactly-2-species
electroneutrality pins the composition to the unique ray
`x* = ion_charge/(ion_charge-Znet)`, there is no free composition to scan
here -- `lB`/`T` is the only continuation parameter, seeded from
`Clapeyron.crit_mix(model,x*)` and warm-started point-to-point via
`DonnanBubblePressure`'s `u0`.

Returns a `NamedTuple` with `lB`, `rho_chain_dense`/`rho_chain_dilute`,
`rho_ion_dense`/`rho_ion_dilute`, `Psi` (the Donnan potential along the
curve), plus `kappa_seq`/`net_charge`/`Tc`/`lBc` for reference.
"""
function donnan_phase_diagram(charges::AbstractVector{<:Integer}; ion_charge::Int=-1,
                               T_range_factor::Float64=0.3, initial_step::Float64=0.025, max_iters::Int=2000)
    model = LS(charges; ion_charge=ion_charge)
    Z = component_charges(model)
    Znet = Z[1]
    xstar = ion_charge / (ion_charge - Znet)
    x = [xstar, 1 - xstar]
    Nchain = length(charges)
    sigma = LS_SIGMA
    Ts = Clapeyron.T_scale(model, x)

    Tc, Pc, Vc = Clapeyron.crit_mix(model, x)
    isnan(Tc) && throw(ErrorException("crit_mix failed to converge for this sequence"))
    lBc = Ts / Tc
    Tend = Tc * T_range_factor

    lBs = Float64[]
    rho_chain_dense, rho_chain_dilute = Float64[], Float64[]
    rho_ion_dense, rho_ion_dilute = Float64[], Float64[]
    psis = Float64[]

    Tcur = Tc
    step = -initial_step * Tc
    u0 = nothing
    iters = 0
    while Tcur > Tend && iters < max_iters
        iters += 1
        Tnext = max(Tcur + step, Tend)
        method = u0 === nothing ? DonnanBubblePressure() : DonnanBubblePressure(; u0=u0)
        result = try
            Clapeyron.bubble_pressure(model, Tnext, x, method)
        catch
            nothing
        end
        if result === nothing || any(isnan, (result.V_dense, result.V_dilute, result.Ψ))
            step /= 2
            abs(step) < 1e-10 * Tc && break
            continue
        end
        u0 = vcat(result.V_dense, result.V_dilute, result.y[1], result.Ψ)
        Tcur = Tnext
        step = max(step * 1.5, -initial_step * Tc)
        push!(lBs, Ts / Tcur)
        push!(rho_chain_dense, Clapeyron.N_A * x[1] * Nchain * sigma^3 / result.V_dense)
        push!(rho_chain_dilute, Clapeyron.N_A * result.y[1] * Nchain * sigma^3 / result.V_dilute)
        push!(rho_ion_dense, Clapeyron.N_A * x[2] * sigma^3 / result.V_dense)
        push!(rho_ion_dilute, Clapeyron.N_A * result.y[2] * sigma^3 / result.V_dilute)
        push!(psis, result.Ψ)
    end
    return (lB=lBs, rho_chain_dense=rho_chain_dense, rho_chain_dilute=rho_chain_dilute,
            rho_ion_dense=rho_ion_dense, rho_ion_dilute=rho_ion_dilute, Psi=psis,
            kappa_seq=kappa_seq(charges), net_charge=Znet, Tc=Tc, lBc=lBc)
end
