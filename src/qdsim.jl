module qdsim

using LinearAlgebra
try
    using MKL
catch e
    @info "MKL is not installed. Using default BLAS implementation: $(BLAS.get_config())."
else
    @info "MKL is loaded."
end

using Comonicon

include("QDSimUtilities.jl")
include("ParseInput.jl")
include("Dynamics.jl")
include("Equilibrium.jl")
include("Simulate.jl")
include("Post.jl")

"""
Quantum dynamics simulations using QuantumDynamics.jl made a breeze
"""
Comonicon.@main

end
