"""
Module for simulating the dynamics of the system.
"""
@cast module Simulate

using Comonicon
using TOML
using QuantumDynamics

using ..QDSimUtilities, ..ParseInput, ..Dynamics, ..Equilibrium

function calc(::QDSimUtilities.Calculation"dynamics", sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, units::QDSimUtilities.Units, sim_node, sim_out::Union{Nothing,HDF5.Group}; dry=false)
    sim.dt = sim_node["dt"] * units.time_unit
    dt_group = Utilities.create_and_select_group(sim_out, "dt=$(sim.dt / units.time_unit)")
    Dynamics.dynamics(QDSimUtilities.Method(sim.method)(), units, sys, bath, sim, dt_group, sim_node; dry)
end

function calc(::QDSimUtilities.Calculation"equilibrium_rho", sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, units::QDSimUtilities.Units, sim_node, sim_out::Union{Nothing,HDF5.Group}; dry=false)
    dat_group = Utilities.create_and_select_group(sim_out, "nsteps=$(sim.nsteps)")
    Equilibrium.rho(QDSimUtilities.Method(sim.method)(), units, sys, bath, sim, dat_group, sim_node; dry)
end

function calc(::QDSimUtilities.Calculation"complex_corr", sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, units::QDSimUtilities.Units, sim_node, sim_out::Union{Nothing, HDF5.Group}; dry=false)
    sim.dt = sim_node["dt"] * units.time_unit
    dat_group = Utilities.create_and_select_group(sim_out, "nsteps=$(sim.nsteps)")
    Equilibrium.complex_time_correlation_function(QDSimUtilities.Method(sim.method)(), units, sys, bath, sim, dat_group, sim_node; dry)
end

@cast function propagate_using_tmats(system_input, simulate_input)
    QDSimUtilities.print_banner()
    @info "Using $(Utilities.get_BLAS_implementation()) for linear algebra."
    @info "Using transfer tensors to propagate the reduced density matrices:"
    QDSimUtilities.print_citation(TTM.references)
    units, sys, bath = ParseInput.parse_system_bath(system_input)
    sim_file = TOML.parsefile(simulate_input)
    prop_node = sim_file["propagate"]
    for (ns, sim_node) in enumerate(prop_node["simulation"])
        @info "Processing simulation number $(ns)."
        sim = ParseInput.parse_sim(sim_node, units)
        @assert isfile(sim.output) "File not present."
        out = h5open(sim.output, "r+")
        method_group = out["$(sim.name)/$(sim.calculation)/$(sim.method)"]
        data_node = calc(QDSimUtilities.Calculation(sim.calculation)(), sys, bath, sim, units, sim_node, method_group; dry=true)
        Ts = read_dataset(data_node, "T0e")
        U0es = TTM.get_propagators_from_Ts(Ts, sim.nsteps)

        ρ0s = sim_node["rho0"]
        outputdirs = sim_node["outgroup"]
        for (nρ, (ρ0file, outputdir)) in enumerate(zip(ρ0s, outputdirs))
            @info "Processing initial density number $(nρ)."
            ρ0 = ParseInput.read_matrix(ρ0file)
            ts, ρs = Utilities.apply_propagator(; propagators=U0es, ρ0, ntimes=sim.nsteps, sim.dt)
            @info "Saving the data in $(outputdir)."
            out = Utilities.create_and_select_group(data_node, outputdir)
            Utilities.check_or_insert_value(out, "time", collect(ts))
            Utilities.check_or_insert_value(out, "time_unit" , units.time_unit)
            Utilities.check_or_insert_value(out, "rho", ρs)
        end
        close(out)
    end
end

@cast function run(system_input, simulate_input)
    QDSimUtilities.print_banner()

    @info "Using $(Utilities.get_BLAS_implementation()) for linear algebra."

    units, sys, bath = ParseInput.parse_system_bath(system_input)
    sim_file = TOML.parsefile(simulate_input)
    sim_node = sim_file["simulation"]
    sim = ParseInput.parse_sim(sim_node, units)

    out = isfile(sim.output) ? h5open(sim.output, "r+") : h5open(sim.output, "w")
    sim_out = Utilities.create_and_select_group(out, sim.name)
    Utilities.check_or_insert_value(sim_out, "Hamiltonian", sys.Hamiltonian / units.energy_unit)
    Utilities.check_or_insert_value(sim_out, "energy_unit", units.energy_unit)
    Utilities.check_or_insert_value(sim_out, "beta", bath.β)
    for (i, jw) in enumerate(bath.Jw)
        ω, j = SpectralDensities.tabulate(jw, false)
        Utilities.check_or_insert_value(sim_out, "bath #$(i)", Matrix{Float64}([ω'; j']))
    end
    calc_group = Utilities.create_and_select_group(sim_out, sim.calculation)
    method_group = Utilities.create_and_select_group(calc_group, sim.method)
    calc(QDSimUtilities.Calculation(sim.calculation)(), sys, bath, sim, units, sim_node, method_group)
    close(out)
end

end
