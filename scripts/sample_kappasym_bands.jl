#=
Targeted fill for the specific holes the user pointed out: kappa_seq > 0.5
combined with kappa_sym near 0.2, 0.5, or 0.8 -- not a fundamental N=20
combinatorial limit (kappa_sym's raw values have resolution ~1/1040), but a
sampling gap in `high_kappa_asymmetric_sequence`'s *default* p_fh choice
(the most-imbalanced first half for a given n_anti), which locks kappa_seq
and kappa_sym together almost one-to-one. Sampling `p_fh` across its full
net-neutral-feasible range (not just its default endpoint) decouples them --
confirmed by direct hit-rate estimation (2000-trial sweeps): ~1%, ~0.5%, and
~2.9% for the three target bands respectively. Low but real, so this script
retries internally (cheap -- pure integer arithmetic, no DFT until a hit is
found) with a generous budget.

Task index cycles through the three target bands (idx%3) so an array
submission spreads roughly evenly across all three.

Usage: julia --project=. scripts/sample_kappasym_bands.jl <task_index>
=#
using IDPInterface
using Clapeyron
using Random
using JLD2

const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "neutral_sample")
mkpath(RESULTS_DIR)

const TARGET_LB = 1.0
const N = 20
const HALF = N ÷ 2
const KAPPA_SEQ_MIN = 0.5
const TOL = 0.05
const TARGETS = (0.2, 0.5, 0.8)
const MAX_ATTEMPTS = 20_000

idx = parse(Int, ARGS[1])
target_ksym = TARGETS[(idx - 1) % 3 + 1]
outpath = joinpath(RESULTS_DIR, "ksb_$(idx).jld2")

rng = Random.MersenneTwister(idx + 3_100_000)

charges = nothing
κ = 0.0
ksym = 0.0
for attempt in 1:MAX_ATTEMPTS
    n_anti = rand(rng, 2:2:8)
    lo, hi = (HALF - n_anti) ÷ 2, (HALF + n_anti) ÷ 2
    p_fh = rand(rng, lo:hi)
    nswaps = rand(rng, 0:1)
    cand = high_kappa_asymmetric_sequence(N, rng; n_anti=n_anti, nswaps=nswaps, p_fh=p_fh)
    ck = kappa_seq(cand)
    cksym = kappa_sym(cand)
    if ck > KAPPA_SEQ_MIN && abs(cksym - target_ksym) < TOL
        global charges = cand
        global κ = ck
        global ksym = cksym
        break
    end
end
charges === nothing && error("task $idx: failed to find a match near kappa_sym=$target_ksym after $MAX_ATTEMPTS attempts")

println("task $idx: target_ksym=$target_ksym  kappa_seq=$(round(κ, digits=3))  kappa_sym=$(round(ksym, digits=3))  charges=$charges")
flush(stdout)

coex = try
    coexistence_at_lB(charges, TARGET_LB)
catch e
    println("task $idx: coexistence_at_lB errored: $e")
    flush(stdout)
    nothing
end

if coex === nothing
    println("task $idx: does not phase-separate at lB=$TARGET_LB (or coexistence search failed)")
    flush(stdout)
    JLD2.jldopen(outpath, "w"; compress=true) do file
        file["charges"] = collect(charges)
        file["kappa_seq"] = κ
        file["phase_separates"] = false
    end
else
    println("task $idx: coexistence found -- rho_dense=$(coex.rho_dense)  rho_dilute=$(coex.rho_dilute)  Tc=$(coex.Tc)")
    flush(stdout)
    model = LS(charges; expand=true)
    point = run_interfacial_point(model, TARGET_LB, coex.rho_dense, coex.rho_dilute)
    JLD2.jldopen(outpath, "w"; compress=true) do file
        file["charges"] = collect(charges)
        file["kappa_seq"] = κ
        file["phase_separates"] = true
        file["lB"] = TARGET_LB
        file["rho_dense"] = coex.rho_dense
        file["rho_dilute"] = coex.rho_dilute
        file["Tc"] = coex.Tc
        file["lBc"] = coex.lBc
        file["gamma"] = point.gamma
        file["x"] = point.x
        file["rho"] = point.rho
        file["Vext"] = point.Vext
    end
    println("task $idx: saved $outpath  gamma=$(point.gamma)")
    flush(stdout)
end
