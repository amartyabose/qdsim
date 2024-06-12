module qdsim

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
@main

end