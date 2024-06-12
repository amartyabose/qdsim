module Equilibrium

using QuantumDynamics
using ..QDSimUtilities

function rho(::QDSimUtilities.Method"TNPI", units::QDSimUtilities.Units, sys::QDSimUtilities.System, bath::QDSimUtilities.Bath, sim::QDSimUtilities.Simulation, dat_group::HDF5.Group, sim_node; dry=false)
    cutoff = get(sim_node, "cutoff", 1e-10)
    maxdim = get(sim_node, "maxdim", 1000)
    algorithm = get(sim_node, "algorithm", "naive")

    maxdim_group = Utilities.create_and_select_group(dat_group, "maxdim=$(maxdim)")
    cutoff_group = Utilities.create_and_select_group(maxdim_group, "cutoff=$(cutoff)")
    data = Utilities.create_and_select_group(cutoff_group, "algorithm=$(algorithm)")
    if !dry
        flush(data)
        extraargs = Utilities.TensorNetworkArgs(; cutoff, maxdim, algorithm)

        nsites = size(sys.Hamiltonian, 1)

        idmat = Matrix(1.0I, nsites, nsites)
        At, _ = ComplexTimePI.A_of_t(; Hamiltonian=sys.Hamiltonian, β=bath.β, t=0.0, N, Jw=bath.Jw, svec=bath.svecs, A=idmat, extraargs)
        Z = tr(At)
        Utilities.check_or_insert_value(data, "eqm_rho", real.(At / Z))
    end
    data
end


end