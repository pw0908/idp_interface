const _rcparams_applied = Ref(false)

_rcparams_tuple() = (plt, WIDTH, DOUBLE_WIDTH, DPI)

"""
    apply_rcparams!()

Apply the project's default PyPlot rcParams (`resources/rcParams.txt`,
included verbatim and unmodified -- it expects `plt`/`PyDict` in scope,
both provided by this module) and return `(plt, WIDTH, DOUBLE_WIDTH, DPI)`.
Idempotent: only actually runs the include the first time it's called.
`include()`-ing a file at runtime bumps Julia's global "world age", so the
lookup of the newly-defined globals is deferred via `invokelatest`.
"""
function apply_rcparams!()
    if !_rcparams_applied[]
        include(joinpath(@__DIR__, "..", "resources", "rcParams.txt"))
        _rcparams_applied[] = true
    end
    return Base.invokelatest(_rcparams_tuple)
end

"""
    save_phase_diagram(path, result)

Save a `phase_diagram` result (a `NamedTuple` with `lB`, `rho_dilute`,
`rho_dense`, `kappa_seq`) to a compressed JLD2 file.
"""
function save_phase_diagram(path::AbstractString, result; charges=nothing)
    JLD2.jldopen(path, "w"; compress=true) do file
        file["lB"] = result.lB
        file["rho_dilute"] = result.rho_dilute
        file["rho_dense"] = result.rho_dense
        file["kappa_seq"] = result.kappa_seq
        file["Tc"] = result.Tc
        file["lBc"] = result.lBc
        charges !== nothing && (file["charges"] = collect(charges))
    end
end

"""
    load_phase_diagram(path)

Load a phase diagram previously saved with `save_phase_diagram`, returning
the same `NamedTuple` shape (plus `charges` if it was saved).
"""
function load_phase_diagram(path::AbstractString)
    JLD2.jldopen(path, "r") do file
        base = (lB=file["lB"], rho_dilute=file["rho_dilute"], rho_dense=file["rho_dense"], kappa_seq=file["kappa_seq"],
                Tc=file["Tc"], lBc=file["lBc"])
        haskey(file, "charges") && return merge(base, (charges=file["charges"],))
        return base
    end
end

"""
    save_donnan_phase_diagram(path, result; charges=nothing)

Save a `donnan_phase_diagram` result (a `NamedTuple` with `lB`,
`rho_chain_dense`/`rho_chain_dilute`, `rho_ion_dense`/`rho_ion_dilute`,
`Psi`, `kappa_seq`, `net_charge`, `Tc`, `lBc`) to a compressed JLD2 file.
"""
function save_donnan_phase_diagram(path::AbstractString, result; charges=nothing)
    JLD2.jldopen(path, "w"; compress=true) do file
        file["lB"] = result.lB
        file["rho_chain_dense"] = result.rho_chain_dense
        file["rho_chain_dilute"] = result.rho_chain_dilute
        file["rho_ion_dense"] = result.rho_ion_dense
        file["rho_ion_dilute"] = result.rho_ion_dilute
        file["Psi"] = result.Psi
        file["kappa_seq"] = result.kappa_seq
        file["net_charge"] = result.net_charge
        file["Tc"] = result.Tc
        file["lBc"] = result.lBc
        charges !== nothing && (file["charges"] = collect(charges))
    end
end

"""
    load_donnan_phase_diagram(path)

Load a Donnan phase diagram previously saved with `save_donnan_phase_diagram`.
"""
function load_donnan_phase_diagram(path::AbstractString)
    JLD2.jldopen(path, "r") do file
        base = (lB=file["lB"], rho_chain_dense=file["rho_chain_dense"], rho_chain_dilute=file["rho_chain_dilute"],
                rho_ion_dense=file["rho_ion_dense"], rho_ion_dilute=file["rho_ion_dilute"], Psi=file["Psi"],
                kappa_seq=file["kappa_seq"], net_charge=file["net_charge"], Tc=file["Tc"], lBc=file["lBc"])
        haskey(file, "charges") && return merge(base, (charges=file["charges"],))
        return base
    end
end

"""
    save_interfacial(path, result)

Save a `run_interfacial_sequence` result to a compressed JLD2 file.
"""
function save_interfacial(path::AbstractString, result)
    JLD2.jldopen(path, "w"; compress=true) do file
        file["charges"] = result.charges
        file["kappa_seq"] = result.kappa_seq
        file["lB"] = result.lB
        file["rho_dense"] = result.rho_dense
        file["rho_dilute"] = result.rho_dilute
        file["gamma"] = result.gamma
        file["x"] = result.x
        file["rho"] = result.rho
        file["Vext"] = result.Vext
    end
end

"""
    load_interfacial(path)

Load a `run_interfacial_sequence` result previously saved with
`save_interfacial`.
"""
function load_interfacial(path::AbstractString)
    JLD2.jldopen(path, "r") do file
        return (charges=file["charges"], kappa_seq=file["kappa_seq"], lB=file["lB"],
                rho_dense=file["rho_dense"], rho_dilute=file["rho_dilute"], gamma=file["gamma"],
                x=file["x"], rho=file["rho"], Vext=file["Vext"])
    end
end

"""
    plot_phase_diagrams(results; labels, path=nothing)

Overlay binodal curves (`lB` vs. reduced density, dilute+dense branches)
for a collection of `phase_diagram` results, one curve per sequence,
labeled by `labels` (typically `kappa_seq` values). Uses the project's
`rcParams.txt` styling. Saves to `path` if given.
"""
function plot_phase_diagrams(results::AbstractVector, labels::AbstractVector; path=nothing)
    plt, WIDTH, _, DPI = apply_rcparams!()
    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH))
    for (result, label) in zip(results, labels)
        isempty(result.lB) && continue
        order = sortperm(result.lB)
        lB = result.lB[order]
        ρd = result.rho_dilute[order]
        ρc = result.rho_dense[order]
        line, = ax.plot(ρc, lB, label=raw"$\kappa=" * string(round(label, digits=2)) * raw"$")
        ax.plot(ρd, lB, color=line.get_color())
    end
    ax.set_xlabel(raw"$\rho^\star$")
    ax.set_ylabel(raw"$l_B/\sigma$")
    ax.legend()
    plt.tight_layout()
    path !== nothing && fig.savefig(path, dpi=DPI)
    return fig
end
