module qdsim

mklext = Base.get_extension(qdsim, :MKLLinearAlgebra)
if !isnothing(mklext)
    @info "Loading MKL for linear algebra."
    using MKL
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
