module ParseInput

using DelimitedFiles
using LinearAlgebra
using TOML
using Unitful, UnitfulAtomic

using QuantumDynamics
using ..QDSimUtilities

function read_matrix(fname::String, mat_type::String="real")
    if mat_type == "real"
        Matrix{ComplexF64}(readdlm(fname))
    elseif mat_type == "complex"
        readdlm(fname, ComplexF64)
    end
end

function parse_unit(input_dict)
    energy_unit = 1.0
    energy_unit_name = "ha"
    time_unit = 1.0
    time_unit_name = "au"

    if haskey(input_dict, "units")
        energy_unit_name = get(input_dict["units"], "energy_unit_name", "ha")
        time_unit_name = get(input_dict["units"], "time_unit_name", "au")

        energy_unit_val = uparse(energy_unit_name)
        if energy_unit_name == "cm^-1"
            energy_unit_val *= Unitful.c * Unitful.h
        end
        time_unit_val = uparse(time_unit_name)

        energy_unit = austrip(1 * energy_unit_val)
        time_unit = austrip(1 * time_unit_val)
    end

    QDSimUtilities.Units(energy_unit, energy_unit_name, time_unit, time_unit_name)
end

function parse_system(sys_inp, unit)
    Htype = get(sys_inp, "Htype", "file")
    H0 = if Htype == "file"
        read_matrix(sys_inp["Hamiltonian"], get(sys_inp, "type", "real")) * unit.energy_unit
    elseif Htype == "nearest_neighbor"
        site_energy = sys_inp["site_energy"] * unit.energy_unit
        coupling = sys_inp["coupling"] * unit.energy_unit
        num_sites = sys_inp["num_sites"]
        Utilities.create_nn_hamiltonian(; site_energies=repeat([site_energy], num_sites), couplings=repeat([coupling], num_sites-1), periodic=false)
    elseif Htype == "nearest_neighbor_cavity"
        site_energy = sys_inp["site_energy"] * unit.energy_unit
        coupling = sys_inp["coupling"] * unit.energy_unit
        num_sites = sys_inp["num_sites"]
        cavity_energy = sys_inp["cavity_energy"] * unit.energy_unit
        cavity_coupling = sys_inp["cavity_coupling"] * unit.energy_unit
        H = Matrix{ComplexF64}(undef, num_sites+1, num_sites+1)
        H[1:num_sites, 1:num_sites] .= Utilities.create_nn_hamiltonian(; site_energies=repeat([site_energy], num_sites), couplings=repeat([coupling], num_sites-1), periodic=false)
        H[end, :] .= cavity_coupling / sqrt(num_sites)
        H[:, end] .= cavity_coupling / sqrt(num_sites)
        H[end, end] = cavity_energy
        H
    end
    ρ0 = nothing
    if haskey(sys_inp, "init_rho")
        ρ0 = read_matrix(sys_inp["init_rho"], "real")
    end
    QDSimUtilities.System(Htype, H0, ρ0)
end

function get_bath(b, unit)
    sd_type = get(b, "type", "ohmic")
    J = nothing
    if sd_type == "ohmic"
        ξ = b["xi"]
        ωc = b["omegac"]
        n = get(b, "n", 1.0)
        Δs = get(b, "Ds", 2.0)
        npoints = get(b, "npoints", 100000)
        ωmax = get(b, "omega_max", 30.0 * ωc)
        ωc *= unit.energy_unit
        ωmax *= unit.energy_unit
        classical = get(b, "classical", false)
        J = SpectralDensities.ExponentialCutoff(; ξ, ωc, n, Δs, ωmax, npoints, classical)
    elseif sd_type == "drude_lorentz"
        λ = b["lambda"] * unit.energy_unit
        γ = b["gamma"]
        ωmax = get(b, "omega_max", 100.0 * γ)
        γ *= unit.energy_unit
        ωmax *= unit.energy_unit
        npoints = get(b, "npoints", 100000)
        Δs = get(b, "Ds", 2.0)
        classical = get(b, "classical", false)
        J = SpectralDensities.DrudeLorentz(; λ, γ, Δs, ωmax, npoints, classical)
    elseif sd_type == "tabular"
        inpfile = b["jw_file"]
        skipstart = get(b, "skipstart", 1)
        classical = get(b, "classical", false)
        J = SpectralDensities.read_jw(inpfile; skipstart, classical)
    elseif sd_type == "tabular_jw_over_w"
        inpfile = b["jw_over_w_file"]
        skipstart = get(b, "skipstart", 1)
        classical = get(b, "classical", false)
        J = SpectralDensities.read_jw(inpfile; skipstart, classical)
    elseif sd_type == "huang_rhys"
        inpfile = b["huang_rhys_file"]
        skipstart = get(b, "skipstart", 1)
        classical = get(b, "classical", false)
        J = SpectralDensities.read_huang_rhys(inpfile; skipstart, classical)
    else
        throw(ArgumentError("Spectral density of type $(sd_type) not supported."))
    end
    J
end

function parse_bath(baths, sys, unit)
    β = 0.0
    if haskey(baths, "beta")
        β = baths["beta"]
    elseif haskey(baths, "temperature")
        β = 1.0 / austrip(baths["temperature"] * uparse("K") * Unitful.k)
    end
    Jw = Vector{SpectralDensities.SpectralDensity}()
    svecs = zeros(size(sys.Hamiltonian, 1), size(sys.Hamiltonian, 1))
    btype = get(baths, "baths_type", "list")
    if btype == "list"
        svecs = zeros(length(baths["bath"]), size(sys.Hamiltonian, 1))
        for (nb, b) in enumerate(baths["bath"])
            push!(Jw, get_bath(b, unit))
            svecs[nb, :] .= b["svec"]
        end
    elseif btype == "site_based"
        bath = get_bath(baths["bath"][1], unit)
        nsites = sys.Htype == "nearest_neighbor_cavity" ? size(sys.Hamiltonian, 1) - 1 : size(sys.Hamiltonian, 1)
        svecs = zeros(nsites, size(sys.Hamiltonian, 1))
        for nb = 1:nsites
            push!(Jw, bath)
            svecs[nb, nb] = 1.0
        end
    end
    QDSimUtilities.Bath(β, Jw,svecs)
end

function parse_system_bath(input_file)
    input_dict = TOML.parsefile(input_file)
    unit = parse_unit(input_dict)
    sys = parse_system(input_dict["system"], unit)
    bath = parse_bath(input_dict["baths"], sys, unit)
    is_QuAPI = get(input_dict["system"], "is_QuAPI", true)
    if !is_QuAPI
        sys.Hamiltonian .-= diagm(sum([SpectralDensities.reorganization_energy(j) * bath.svecs[nb, :] .^ 2 for (nb, j) in enumerate(bath.Jw)]))
    end
    unit, sys, bath
end

function parse_sim(sim, unit)
    name = sim["name"]
    calculation = get(sim, "calculation", "dynamics")
    method = sim["method"]
    output = sim["output"]
    nsteps = sim["nsteps"]
    dt = get(sim, "dt", 0.0)
    QDSimUtilities.Simulation(name, calculation, method, output, dt, nsteps)
end

function parse_operator(op, Hamiltonian)
    obs = zero(Hamiltonian)
    if startswith(op, "P_")
        state = parse(Int64, split(op, "_")[2])
        obs[state, state] = 1
        obs
    elseif startswith(op, "F_")
        state = parse(Int64, split(op, "_")[2])
        obs[state, state] = 1
        1im * Utilities.commutator(Hamiltonian, obs)
    else
        ParseInput.read_matrix(op)
    end
end

end
