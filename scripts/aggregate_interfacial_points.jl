#=
Merge per-point results from `scripts/run_interfacial_single_point.jl`
(saved under `results/interfacial_points/<name>_pt<idx>.jld2`) into the
same aggregate shape `save_interfacial` produces
(`results/interfacial_<name>.jld2`), for each sequence in turn.

Usage: julia --project=. scripts/aggregate_interfacial_points.jl
=#
using IDPInterface
using JLD2

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const POINTS_DIR = joinpath(RESULTS_DIR, "interfacial_points")

sequence_names = ["block1", "block2", "block5", "block10", "asym1", "asym2", "asym3"]

for name in sequence_names
    files = filter(f -> occursin(Regex("^$(name)_pt\\d+\\.jld2\$"), basename(f)),
                    readdir(POINTS_DIR; join=true))
    isempty(files) && (println("no points found for $name, skipping"); continue)

    points = [JLD2.load(f) for f in files]
    order = sortperm([p["lB"] for p in points])
    points = points[order]

    charges = points[1]["charges"]
    kappa = points[1]["kappa_seq"]
    lB = [p["lB"] for p in points]
    rho_dense = [p["rho_dense"] for p in points]
    rho_dilute = [p["rho_dilute"] for p in points]
    gamma = [p["gamma"] for p in points]
    x = points[1]["x"]
    ngrid = length(x)
    N = length(charges)
    npts = length(points)
    rho = Array{Float64}(undef, npts, ngrid, N)
    Vext = Array{Float64}(undef, npts, ngrid)
    for (i, p) in enumerate(points)
        rho[i, :, :] = p["rho"]
        Vext[i, :] = p["Vext"]
    end

    result = (charges=charges, kappa_seq=kappa, lB=lB, rho_dense=rho_dense,
              rho_dilute=rho_dilute, gamma=gamma, x=x, rho=rho, Vext=Vext)
    save_interfacial(joinpath(RESULTS_DIR, "interfacial_$(name).jld2"), result)
    println(name, ": aggregated ", npts, " points -> results/interfacial_$(name).jld2")
end
println("done")
