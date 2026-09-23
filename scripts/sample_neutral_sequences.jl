#=
Step 3 hypothesis test at fixed coupling strength: sample many net-neutral
20-mer sequences spanning a wide range of kappa_seq, and for each one that
actually phase-separates at lB=1.0, run a single interfacial calculation
there (density + electrostatic potential profiles, interfacial tension).

Designed to run as one independent SLURM array task per sequence (see
`submit_neutral_sample.sh`) rather than a serial loop -- each task is
seeded deterministically from its array index so the whole sample is
reproducible, and results are written to their own file so tasks never
contend with each other.

Usage: julia --project=. scripts/sample_neutral_sequences.jl <task_index>
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
outpath = joinpath(RESULTS_DIR, "seq_$(idx).jld2")

rng = Random.MersenneTwister(idx)
max_block = rand(rng, 2:10)
charges = random_blocky_sequence(N, rng; max_block=max_block)
κ = kappa_seq(charges)

println("task $idx: max_block=$max_block  kappa_seq=$(round(κ, digits=3))  charges=$charges")
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
