"""
Various routines for post-processing and analysing the simulation results.
"""
@cast module Post

using Comonicon
using QuantumDynamics
using DelimitedFiles
using TOML
using ..ParseInput, ..Simulate, ..QDSimUtilities

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

@cast function get_observable(system_input, simulate_input)
    QDSimUtilities.print_banner()
    units, sys, bath = ParseInput.parse_system_bath(system_input)
    sim_file = TOML.parsefile(simulate_input)
    for (ns, sim_node) in enumerate(sim_file["simulation"])
        @info "Getting observables for simulation number $(ns)."
        sim = ParseInput.parse_sim(sim_node, units)
        @assert isfile(sim.output) "File not present."
        out = h5open(sim.output, "r+")
        outputdir = sim_node["outgroup"]
        method_group = out["$(sim.name)/$(sim.calculation)/$(sim.method)"]
        data_node = Simulate.calc(QDSimUtilities.Calculation(sim.calculation)(), sys, bath, sim, units, sim_node, method_group; dry=true)[outputdir]
        ts = read_dataset(data_node, "ts")
        ρs = read_dataset(data_node, "rhos")
        num_obs = length(sim_node["observable"])
        names = String[]
        vals = zeros(ComplexF64, length(ts), num_obs)
        for (os, obs) in enumerate(sim_node["observable"])
            push!(names, obs["observable"])
            if obs["observable"] == "trace"
                vals[:, os] .= [tr(ρs[j, :, :]) for j in axes(ρs, 1)]
            elseif obs["observable"] == "purity"
                vals[:, os] .= [tr(ρs[j, :, :] * ρs[j, :, :]) for j in axes(ρs, 1)]
            elseif obs["observable"] == "vonNeumann_entropy"
                vals[:, os] = [-tr(ρs[j, :, :] * log(ρs[j, :, :])) for j in axes(ρs, 1)]
            else
                obs = ParseInput.read_matrix(obs["observable"])
                vals[:, os] = Utilities.expect(ρs, obs)
            end
        end

        open("real_"*sim_node["observable_output"], "w") do io
            write(io, "# (1)t ")
            for (j, n) in enumerate(names)
                write(io, "($(j+1))$(n) ")
            end
            write(io, "\n")
            writedlm(io, [round.(ts; sigdigits=10) round.(real.(vals); sigdigits=10)])
        end

        open("imag_"*sim_node["observable_output"], "w") do io
            write(io, "# (1)t ")
            for (j, n) in enumerate(names)
                write(io, "($(j+1))$(n) ")
            end
            write(io, "\n")
            writedlm(io, [round.(ts; sigdigits=10) round.(imag.(vals); sigdigits=10)])
        end
    end
end

end