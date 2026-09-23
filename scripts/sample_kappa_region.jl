#=
Targeted fill for kappa_seq > 0.6 AND kappa_sym > 0.2 jointly, per explicit
user request. Uses `high_kappa_asymmetric_sequence` with `n_anti` drawn from
{0,2,4,6} and `nswaps=0` -- a parameter sweep (300 trials each) found these
give hit rates of 100%/39%/38%/58% for this exact region, far better than
`nswaps>=1` (which rapidly drops to a few percent) or `perturbed_diblock_sequence`
(similar). Retries internally until a hit is found, so every SLURM array
task produces a useful point.

Usage: julia --project=. scripts/sample_kappa_region.jl <task_index>
=#
using IDPInterface
using Clapeyron
using Random
using JLD2

const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "neutral_sample")
mkpath(RESULTS_DIR)

const TARGET_LB = 1.0
const N = 20
const KAPPA_SEQ_MIN = 0.6
const KAPPA_SYM_MIN = 0.2
const MAX_ATTEMPTS = 500

idx = parse(Int, ARGS[1])
outpath = joinpath(RESULTS_DIR, "kreg_$(idx).jld2")

rng = Random.MersenneTwister(idx + 2_300_000)

charges = nothing
κ = 0.0
ksym = 0.0
for attempt in 1:MAX_ATTEMPTS
    n_anti = rand(rng, (0, 2, 4, 6))
    cand = high_kappa_asymmetric_sequence(N, rng; n_anti=n_anti, nswaps=0)
    ck = kappa_seq(cand)
    cksym = kappa_sym(cand)
    if ck > KAPPA_SEQ_MIN && cksym > KAPPA_SYM_MIN
        global charges = cand
        global κ = ck
        global ksym = cksym
        break
    end
end
charges === nothing && error("task $idx: failed to find a match after $MAX_ATTEMPTS attempts")

println("task $idx: kappa_seq=$(round(κ, digits=3))  kappa_sym=$(round(ksym, digits=3))  charges=$charges")
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
