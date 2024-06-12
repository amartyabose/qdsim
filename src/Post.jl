"""
Various routines for post-processing and analysing the simulation results.
"""
@cast module Post

using Comonicon
using QuantumDynamics
using DelimitedFiles
using TOML
using ..ParseInput, ..Simulate, ..QDSimUtilities
include("plot_settings.jl")

"""
Combine the source files `sources` into `output`. If `output` does not exist, it is created.

# Args
- `sources`: source output files
- `output`: destination output file
"""
@cast function merge_into(sources::String...; output::String)
    for input in sources
        Utilities.merge_into(input, output)
    end
end

@cast function plot(system_input, simulate_input)
    QDSimUtilities.print_banner()
    units, sys, bath = ParseInput.parse_system_bath(system_input)
    sim_file = TOML.parsefile(simulate_input)
    plot_node = sim_file["plot"]
    outfile = plot_node["output"]
    xlabel = plot_node["xlabel"]
    ylabel = plot_node["ylabel"]
    new_figure()
    for (ns, sim_node) in enumerate(plot_node["simulation"])
        @info "Plotting for simulation number $(ns)."
        sim = ParseInput.parse_sim(sim_node, units)
        @assert isfile(sim.output) "File not present."
        out = h5open(sim.output, "r+")
        outputdir = sim_node["outgroup"]
        method_group = out["$(sim.name)/$(sim.calculation)/$(sim.method)"]
        data_node = Simulate.calc(QDSimUtilities.Calculation(sim.calculation)(), sys, bath, sim, units, sim_node, method_group; dry=true)[outputdir]
        ts = read_dataset(data_node, "ts")
        ρs = read_dataset(data_node, "rhos")
        observable = sim_node["observable"]
        if observable == "trace"
            vals = [tr(ρs[j, :, :]) for j in axes(ρs, 1)]
        elseif observable == "purity"
            vals = [tr(ρs[j, :, :] * ρs[j, :, :]) for j in axes(ρs, 1)]
        elseif observable == "vonNeumann_entropy"
            vals = [-tr(ρs[j, :, :] * log(ρs[j, :, :])) for j in axes(ρs, 1)]
        else
            obs = ParseInput.read_matrix(observable)
            vals = Utilities.expect(ρs, obs)
        end
        lab = sim_node["label"]
        plt.plot(ts, real.(vals), lw=0.75, label=lab)
        close(out)
    end
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(outfile; bbox_inches="tight")
end

end