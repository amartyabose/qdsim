module QDSimUtilities
using QuantumDynamics

struct Method{method} end
Method(m) = Method{Symbol(m)}
macro Method_str(s)
    return :(Method{$(Expr(:quote, Symbol(s)))})
end

struct Calculation{calculation} end
Calculation(c) = Calculation{Symbol(c)}
macro Calculation_str(c)
    return :(Calculation{$(Expr(:quote, Symbol(c)))})
end

struct Units
    energy_unit::Float64
    energy_unit_name::String
    time_unit::Float64
    time_unit_name::String
end

struct System
    Hamiltonian::Matrix{ComplexF64}
    ρ0::Union{Nothing,Matrix{ComplexF64}}
end

struct Bath
    β::Float64
    Jw::Vector{SpectralDensities.SpectralDensity}
    svecs::Matrix{Float64}
end

struct Simulation
    name::String
    calculation::String
    method::String
    output::String

    dt::Float64
    ntimes::Int64
end

print_citation(cite) = printstyled(cite * "\n"; color=:red)

function print_banner()
    printstyled("Welcome to the QDSim program built on top of the QuantumDynamics.jl library:\n"; color=:blue)
    print_citation("- Bose, A. QuantumDynamics.jl: A Modular Approach to Simulations of Dynamics of Open Quantum Systems. The Journal of Chemical Physics 2023, 158 (20), 204113. https://doi.org/10.1063/5.0151483.")
end
end