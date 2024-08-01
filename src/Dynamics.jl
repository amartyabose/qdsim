module Dynamics

using LinearAlgebra
using QuantumDynamics
using ..QDSimUtilities

function dynamics(::QDSimUtilities.Method"TNPI-TTM", units::QDSimUtilities.Units, sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, dt_group::Union{Nothing,HDF5.Group}, sim_node; dry=false)
    if !dry
        @info "Running a TEMPO / TNPI dynamics calculation with TTM. Please cite:"
        QDSimUtilities.print_citation(TEMPO.references)
        QDSimUtilities.print_citation(TTM.references)
    end
    rmax = sim_node["rmax"]
    rmax_group = Utilities.create_and_select_group(dt_group, "rmax=$(rmax)")
    kmax::Union{Nothing,Int} = get(sim_node, "kmax", nothing)
    cutoff = get(sim_node, "cutoff", 1e-10)
    maxdim = get(sim_node, "maxdim", 1000)
    algorithm = get(sim_node, "algorithm", "naive")
    @info "Running with $(BLAS.get_num_threads()) threads."

    if !isnothing(kmax)
        @assert kmax <= rmax "kmax = $(kmax) should be less than rmax = $(rmax)."
        rmax_group = Utilities.create_and_select_group(rmax_group, "kmax=$(kmax)")
    end
    maxdim_group = Utilities.create_and_select_group(rmax_group, "maxdim=$(maxdim)")
    cutoff_group = Utilities.create_and_select_group(maxdim_group, "cutoff=$(cutoff)")
    data = Utilities.create_and_select_group(cutoff_group, "algorithm=$(algorithm)")

    if !dry
        Utilities.check_or_insert_value(data, "dt", sim.dt / units.time_unit)
        Utilities.check_or_insert_value(data, "time_unit", units.time_unit)
        Utilities.check_or_insert_value(data, "time", 0:sim.dt/units.time_unit:sim.nsteps*sim.dt/units.time_unit |> collect)
        flush(data)
        path_integral_routine = TEMPO.build_augmented_propagator
        extraargs = TEMPO.TEMPOArgs(; cutoff, maxdim, algorithm)
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=sys.Hamiltonian, dt=sim.dt, ntimes=rmax)
        Utilities.check_or_insert_value(data, "fbU", fbU)
        flush(data)
        TTM.get_propagators(; fbU, Jw=bath.Jw, β=bath.β, dt=sim.dt, ntimes=sim.nsteps, rmax, kmax, path_integral_routine, extraargs, svec=bath.svecs, verbose=true, output=data)
        @info "After this run, please run a propagate-using-tmats calculation to obtain the time evolution of a particular density matrix."
    end
    data
end

function dynamics(::QDSimUtilities.Method"QuAPI-TTM", units::QDSimUtilities.Units, sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, dt_group::Union{Nothing,HDF5.Group}, sim_node; dry=false)
    if !dry
        @info "Running a QuAPI calculation with TTM. Please cite:"
        QDSimUtilities.print_citation(QuAPI.references)
        QDSimUtilities.print_citation(TTM.references)
    end
    rmax = sim_node["rmax"]
    rmax_group = Utilities.create_and_select_group(dt_group, "rmax=$(rmax)")
    cutoff = get(sim_node, "cutoff", 1e-10)
    data = Utilities.create_and_select_group(rmax_group, "cutoff=$(cutoff)")
    exec = get(sim_node, "exec", "ThreadedEx")
    if exec != "SequentialEx"
        @info "Running with $(Threads.nthreads()) threads."
    end
    if !dry
        Utilities.check_or_insert_value(data, "dt", sim.dt / units.time_unit)
        Utilities.check_or_insert_value(data, "time_unit", units.time_unit)
        Utilities.check_or_insert_value(data, "time", 0:sim.dt:sim.nsteps*sim.dt |> collect)
        flush(data)

        path_integral_routine = QuAPI.build_augmented_propagator
        extraargs = QuAPI.QuAPIArgs(; cutoff)
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=sys.Hamiltonian, dt=sim.dt, ntimes=rmax)
        Utilities.check_or_insert_value(data, "fbU", fbU)
        flush(data)
        TTM.get_propagators(; fbU, Jw=bath.Jw, β=bath.β, dt=sim.dt, ntimes=sim.nsteps, rmax=rmax, path_integral_routine, extraargs, svec=bath.svecs, verbose=true, output=data, exec=QDSimUtilities.parse_exec(exec))..., data
        @info "After this run, please run a propagate-using-tmats calculation to obtain the time evolution of a particular density matrix."
    end
    data
end

function dynamics(::QDSimUtilities.Method"Blip-TTM", units::QDSimUtilities.Units, sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, dt_group::Union{Nothing,HDF5.Group}, sim_node; dry=false)
    if !dry
        @info "Running a Blip calculation with TTM. Please cite:"
        QDSimUtilities.print_citation(Blip.references)
        QDSimUtilities.print_citation(TTM.references)
    end
    rmax = sim_node["rmax"]
    rmax_group = Utilities.create_and_select_group(dt_group, "rmax=$(rmax)")
    max_blips = get(sim_node, "max_blips", -1)
    min_dist = get(sim_node, "min_dist", 1)
    if max_blips == -1
        mb_data = Utilities.create_and_select_group(rmax_group, "max_blips=all")
    else
        mb_data = Utilities.create_and_select_group(rmax_group, "max_blips=$(max_blips)")
    end
    data = Utilities.create_and_select_group(mb_data, "min_dist=$(min_dist)")
    exec = get(sim_node, "exec", "ThreadedEx")
    if exec != "SequentialEx"
        @info "Running with $(Threads.nthreads()) threads."
    end
    if !dry
        Utilities.check_or_insert_value(data, "dt", sim.dt / units.time_unit)
        Utilities.check_or_insert_value(data, "time_unit", units.time_unit)
        Utilities.check_or_insert_value(data, "time", 0:sim.dt:sim.nsteps*sim.dt |> collect)
        flush(data)

        path_integral_routine = Blip.build_augmented_propagator
        extraargs = Blip.BlipArgs(; max_blips, min_dist)
        fbU = Propagators.calculate_bare_propagators(; Hamiltonian=sys.Hamiltonian, dt=sim.dt, ntimes=rmax)
        Utilities.check_or_insert_value(data, "fbU", fbU)
        flush(data)
        TTM.get_propagators(; fbU, Jw=bath.Jw, β=bath.β, dt=sim.dt, ntimes=sim.nsteps, rmax=rmax, path_integral_routine, extraargs, svec=bath.svecs, verbose=true, output=data, exec=QDSimUtilities.parse_exec(exec))..., data
    end
    data
end

function dynamics(::QDSimUtilities.Method"Forster", units::QDSimUtilities.Units, sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, dt_group::Union{Nothing,HDF5.Group}, sim_node; dry=false)
    if !dry
        @info "Running a Forster calculation. Please cite:"
    end
    data = Utilities.create_and_select_group(dt_group, "Forster")
    if !dry
        Utilities.check_or_insert_value(data, "dt", sim.dt / units.time_unit)
        Utilities.check_or_insert_value(data, "time_unit", units.time_unit)
        Utilities.check_or_insert_value(data, "time", 0:sim.dt:sim.nsteps*sim.dt |> collect)
        flush(data)

        k, U = Forster.build_incoherent_propagator(; H=sys.Hamiltonian, Jw=bath.Jw, dt, β=bath.β, verbose=true)
        Utilities.check_or_insert_value(data, "k", k)
        Utilities.check_or_insert_value(data, "U", U)
    end
    data
end

end