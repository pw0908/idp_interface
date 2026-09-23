#=
Step 3 deliverable: 1D interfacial calculations along each sequence's Step 1
binodal. For every sequence with a saved `results/phasediagram_<name>.jld2`
(the symmetric `block*` family plus the asymmetric `asym*` sequences), runs
`run_interfacial_sequence` at several points spanning the binodal, saving
density/potential profiles and interfacial tension to
`results/interfacial_<name>.jld2`, plus a summary figure (IFT vs. lB,
overlaid across all sequences) and one representative electrostatic
potential profile per sequence.

Usage: julia --project=. scripts/run_interfacial.jl
=#
using IDPInterface
using PyPlot
const PyDict = PyPlot.PyDict

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

sequence_names = ["block1", "block2", "block5", "block10", "asym1", "asym2", "asym3"]

all_results = Dict{String,Any}()

for name in sequence_names
    pd_path = joinpath(RESULTS_DIR, "phasediagram_$(name).jld2")
    isfile(pd_path) || (println("skipping $name: no phase diagram at $pd_path"); continue)
    pd = load_phase_diagram(pd_path)
    charges = pd.charges
    println("=== $name (κ_seq=$(round(kappa_seq(charges), digits=3))) ===")
    flush(stdout)

    result = run_interfacial_sequence(charges, pd; npoints=5)
    save_interfacial(joinpath(RESULTS_DIR, "interfacial_$(name).jld2"), result)
    all_results[name] = result
    println("  saved results/interfacial_$(name).jld2 (", length(result.lB), " points)")
    flush(stdout)
end

# ── Summary figure 1: interfacial tension vs. lB, all sequences overlaid ──
include(joinpath(@__DIR__, "..", "resources", "rcParams.txt"))

fig1, ax1 = subplots()
for (name, result) in all_results
    order = sortperm(result.lB)
    label = "$(name) (" * raw"$\kappa=" * string(round(result.kappa_seq, digits=2)) * raw"$)"
    ax1.plot(result.lB[order], result.gamma[order], marker="o", label=label)
end
ax1.set_xlabel(L"\ell_B/\sigma")
ax1.set_ylabel(L"\gamma^\star")
ax1.legend(fontsize=9)
tight_layout()
savefig(joinpath(RESULTS_DIR, "interfacial_tension_summary.pdf"))
savefig(joinpath(RESULTS_DIR, "interfacial_tension_summary.png"), dpi=200)
println("saved results/interfacial_tension_summary.png/.pdf")

# ── Summary figure 2: IFT vs. kappa_seq at matched density ratio ──────────
fig2, ax2 = subplots()
for (name, result) in all_results
    ratio = result.rho_dense ./ result.rho_dilute
    # pick the point closest to a representative ratio of 1e3 for cross-sequence comparability
    i = argmin(abs.(log.(ratio) .- log(1e3)))
    ax2.scatter([result.kappa_seq], [result.gamma[i]], s=60)
    ax2.annotate(name, (result.kappa_seq, result.gamma[i]), fontsize=8, xytext=(3, 3), textcoords="offset points")
end
ax2.set_xlabel(L"\kappa_{\mathrm{seq}}")
ax2.set_ylabel(L"\gamma^\star \ (\rho_{\mathrm{dense}}/\rho_{\mathrm{dilute}} \approx 10^3)")
tight_layout()
savefig(joinpath(RESULTS_DIR, "interfacial_tension_vs_kappa.pdf"))
savefig(joinpath(RESULTS_DIR, "interfacial_tension_vs_kappa.png"), dpi=200)
println("saved results/interfacial_tension_vs_kappa.png/.pdf")

# ── Summary figure 3: representative electrostatic potential profile per sequence ──
fig3, ax3 = subplots()
for (name, result) in all_results
    ratio = result.rho_dense ./ result.rho_dilute
    i = argmin(abs.(log.(ratio) .- log(1e3)))
    rho_tot = vec(sum(result.rho[i, :, :], dims=2))
    mid_level = sqrt(maximum(rho_tot) * minimum(rho_tot))
    right_half = findall(result.x .> 0)
    icross = right_half[argmin(abs.(rho_tot[right_half] .- mid_level))]
    xc = result.x[icross]
    xrel = result.x .- xc
    label = "$(name) (" * raw"$\kappa=" * string(round(result.kappa_seq, digits=2)) * raw"$)"
    ax3.plot(xrel, result.Vext[i, :], label=label)
end
ax3.set_xlim(-15, 15)
ax3.set_xlabel(L"(x-x_{\mathrm{interface}})/\sigma")
ax3.set_ylabel(L"e\psi/k_BT")
ax3.axhline(0, color="black", lw=0.5, ls=":")
ax3.legend(fontsize=9)
tight_layout()
savefig(joinpath(RESULTS_DIR, "potential_profiles_summary.pdf"))
savefig(joinpath(RESULTS_DIR, "potential_profiles_summary.png"), dpi=200)
println("saved results/potential_profiles_summary.png/.pdf")

println("Step 3 batch run complete.")
