#=
Fills the high-kappa_seq / full-kappa_sym-range region of parameter space
that `sample_neutral_sequences.jl`'s purely random generator essentially
never reaches by chance (per user request, after inspecting
`results/neutral_sample_scatter_kappa_kappasym.png`). Uses
`high_kappa_asymmetric_sequence` (constructive: adapts the first half's own
composition to `n_anti` so both the kappa_sym-maximizing palindrome and the
exact diblock are reachable) instead of `random_blocky_sequence`. Otherwise
identical to `sample_neutral_sequences.jl` -- one independent SLURM array
task per sequence, same lB=1.0 target, same output shape (so the existing
plotting scripts pick these up automatically alongside `seq_*.jld2`).

Usage: julia --project=. scripts/sample_high_kappa_sequences.jl <task_index>
=#
using IDPInterface
using Clapeyron
using Random
using JLD2

const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "neutral_sample")
mkpath(RESULTS_DIR)

const TARGET_LB = 1.0
const N = 20

idx = parse(Int, ARGS[1])
outpath = joinpath(RESULTS_DIR, "hik_$(idx).jld2")

rng = Random.MersenneTwister(idx + 900000)
n_anti = rand(rng, 0:2:10)
nswaps = rand(rng, 0:3)
charges = high_kappa_asymmetric_sequence(N, rng; n_anti=n_anti, nswaps=nswaps)
κ = kappa_seq(charges)
ksym = kappa_sym(charges)

println("task $idx: n_anti=$n_anti nswaps=$nswaps  kappa_seq=$(round(κ, digits=3))  kappa_sym=$(round(ksym, sigdigits=4))  charges=$charges")
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
