#=
Targeted fill for kappa_seq > 0.75, per explicit user request: start from
the exact diblock and only slightly disorder it (`perturbed_diblock_sequence`),
rather than `high_kappa_asymmetric_sequence`'s from-scratch construction.
Retries internally (different nswaps/seed draws) until a kappa_seq>0.75
sequence is found, so every SLURM array task produces a useful point instead
of wasting slots on rejects (empirically, nswaps=1 succeeds ~26% of the
time, nswaps=2 ~8% -- a handful of retries per task is enough in practice).

Usage: julia --project=. scripts/sample_diblock_perturbation.jl <task_index>
=#
using IDPInterface
using Clapeyron
using Random
using JLD2

const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "neutral_sample")
mkpath(RESULTS_DIR)

const TARGET_LB = 1.0
const N = 20
const KAPPA_MIN = 0.75
const MAX_ATTEMPTS = 500

idx = parse(Int, ARGS[1])
outpath = joinpath(RESULTS_DIR, "dib_$(idx).jld2")

rng = Random.MersenneTwister(idx + 1_700_000)

charges = nothing
κ = 0.0
for attempt in 1:MAX_ATTEMPTS
    nswaps = rand(rng, 1:2)  # nswaps>=3 rarely clears kappa_seq>0.75, not worth the attempts
    cand = perturbed_diblock_sequence(N, rng; nswaps=nswaps)
    ck = kappa_seq(cand)
    if ck > KAPPA_MIN
        global charges = cand
        global κ = ck
        break
    end
end
charges === nothing && error("task $idx: failed to find kappa_seq>$KAPPA_MIN after $MAX_ATTEMPTS attempts")

ksym = kappa_sym(charges)
println("task $idx: kappa_seq=$(round(κ, digits=3))  kappa_sym=$(round(ksym, sigdigits=4))  charges=$charges")
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
