#=
Finest-grained Step 3 unit of work: ONE (sequence, binodal point) pair,
designed to run as one SLURM array task among many (see
`submit_interfacial_points.sh`) so a large per-sequence point count (e.g.
100, for a clean interfacial-tension curve) completes in wall-clock time
comparable to a single point, not 100x that.

Usage: julia --project=. scripts/run_interfacial_single_point.jl <seqname> <npoints> <point_index>

`point_index` (1-based) indexes into `select_binodal_indices(pd; npoints)`'s
deterministic list -- the same list `run_interfacial_sequence` would use if
it ran all `npoints` serially in one job.
=#
using IDPInterface
using Clapeyron
using JLD2

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const POINTS_DIR = joinpath(RESULTS_DIR, "interfacial_points")
mkpath(POINTS_DIR)

name = ARGS[1]
npoints = parse(Int, ARGS[2])
point_index = parse(Int, ARGS[3])

pd_path = joinpath(RESULTS_DIR, "phasediagram_$(name).jld2")
isfile(pd_path) || error("no phase diagram at $pd_path")
pd = load_phase_diagram(pd_path)
charges = pd.charges

idxs = select_binodal_indices(pd; npoints=npoints)
if point_index > length(idxs)
    println("point_index $point_index exceeds $(length(idxs)) available (deduplicated) points for $name -- nothing to do")
    exit(0)
end
idx = idxs[point_index]
lB = pd.lB[idx]
rho_dense_star = pd.rho_dense[idx]
rho_dilute_star = pd.rho_dilute[idx]

outpath = joinpath(POINTS_DIR, "$(name)_pt$(point_index).jld2")
println("$name pt$point_index/$( length(idxs) ): lB=$(round(lB,digits=4))  ratio=$(round(rho_dense_star/rho_dilute_star,sigdigits=3))")
flush(stdout)

model = LS(charges; expand=true)
point = run_interfacial_point(model, lB, rho_dense_star, rho_dilute_star)

JLD2.jldopen(outpath, "w"; compress=true) do file
    file["charges"] = collect(charges)
    file["kappa_seq"] = kappa_seq(charges)
    file["lB"] = lB
    file["rho_dense"] = rho_dense_star
    file["rho_dilute"] = rho_dilute_star
    file["gamma"] = point.gamma
    file["x"] = point.x
    file["rho"] = point.rho
    file["Vext"] = point.Vext
end
println("saved $outpath  gamma=$(point.gamma)")
