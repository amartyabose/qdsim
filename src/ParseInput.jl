module ParseInput

using DelimitedFiles
using TOML
using Unitful, UnitfulAtomic

using QuantumDynamics
using ..QDSimUtilities

function read_matrix(fname::String, mat_type::String="real")
    if mat_type == "real"
        Matrix{ComplexF64}(readdlm(fname))
    elseif mat_type == "complex"
        readdlm(fname, ' ', ComplexF64)
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
    H0 = read_matrix(sys_inp["Hamiltonian"], get(sys_inp, "type", "real")) * unit.energy_unit
    ρ0 = nothing
    if haskey(sys_inp, "init_rho")
        ρ0 = read_matrix(sys_inp["init_rho"], "real")
    end
    QDSimUtilities.System(H0, ρ0)
end

function parse_bath(baths, H0, unit)
    β = 0.0
    if haskey(baths, "beta")
        β = baths["beta"]
    elseif haskey(baths, "temperature")
        β = 1.0 / austrip(baths["temperature"] * uparse("K") * Unitful.k)
    end
    Jw = Vector{SpectralDensities.SpectralDensity}()
    svecs = zeros(length(baths["bath"]), size(H0, 1))
    for (nb, b) in enumerate(baths["bath"])
        sd_type = get(b, "type", "ohmic")
        if sd_type == "ohmic"
            ξ = b["xi"]
            ωc = b["omegac"] * unit.energy_unit
            n = get(b, "n", 1.0)
            Δs = get(b, "Ds", 2.0)
            npoints = get(b, "npoints", 100000)
            ωmax = get(b, "omega_max", 30.0 * ωc)
            classical = get(b, "classical", false)
            push!(Jw, SpectralDensities.ExponentialCutoff(; ξ, ωc, n, Δs, ωmax, npoints, classical))
        elseif sd_type == "drude_lorentz"
            λ = b["lambda"] * unit.energy_unit
            γ = b["gamma"] * unit.energy_unit
            ωmax = get(b, "omega_max", 100.0 * γ)
            npoints = get(b, "npoints", 100000)
            Δs = get(b, "Ds", 2.0)
            classical = get(b, "classical", false)
            push!(Jw, SpectralDensities.DrudeLorentz(; λ, γ, Δs, ωmax, npoints, classical))
        elseif sd_type == "tabular"
            inpfile = b["jw_file"]
            skipstart = get(b, "skipstart", 1)
            classical = get(b, "classical", false)
            push!(Jw, SpectralDensities.read_jw(inpfile; skipstart, classical))
        elseif sd_type == "tabular_jw_over_w"
            inpfile = b["jw_over_w_file"]
            skipstart = get(b, "skipstart", 1)
            classical = get(b, "classical", false)
            push!(Jw, SpectralDensities.read_jw(inpfile; skipstart, classical))
        elseif sd_type == "huang_rhys"
            inpfile = b["huang_rhys_file"]
            skipstart = get(b, "skipstart", 1)
            classical = get(b, "classical", false)
            push!(Jw, SpectralDensities.read_huang_rhys(inpfile; skipstart, classical))
        else
            throw(ArgumentError("Spectral density of type $(sd_type) not supported."))
        end
        svecs[nb, :] .= b["svec"]
    end
    QDSimUtilities.Bath(β, Jw, svecs)
end

function parse_system_bath(input_file)
    input_dict = TOML.parsefile(input_file)
    unit = parse_unit(input_dict)
    sys = parse_system(input_dict["system"], unit)
    bath = parse_bath(input_dict["baths"], sys.Hamiltonian, unit)
    is_QuAPI = get(input_dict["system"], "is_QuAPI", true)
    if !is_QuAPI
        sys.Hamiltonian .-= diagm(sum([SpectralDensities.reorganization_energy(j) * svec[nb, :] .^ 2 for (nb, j) in enumerate(Jw)]))
    end
    unit, sys, bath
end

function parse_sim(sim, unit)
    name = sim["name"]
    calculation = get(sim, "calcation", "dynamics")
    method = sim["method"]
    output = sim["output"]
    ntimes = sim["ntimes"]
    QDSimUtilities.Simulation(name, calculation, method, output, 0.0, ntimes)
end

end