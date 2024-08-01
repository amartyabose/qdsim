module qdsim

using LinearAlgebra
using MKL
@info "Using $(BLAS.get_config()) for linear algebra."

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
