module SpinGlassEngine

using Base: Tuple
using SpinGlassTensors, SpinGlassNetworks
using TensorOperations, TensorCast
using MetaGraphs
using Memoize
using LinearAlgebra
using LightGraphs

SpinGlassNetworks.local_basis(ψ::AbstractMPS, i::Int) = SpinGlassNetworks.local_basis(physical_dim(ψ, i))

function LinearAlgebra.dot(ψ::AbstractMPS, state::Union{Vector, NTuple})
    C = I

    for (M, σ) ∈ zip(ψ, state)
        i = idx(σ)
        C = M[:, i, :]' * (C * M[:, i, :])
    end
    tr(C)
end

include("network_operations.jl")
include("network_interface.jl")
include("MPS_search.jl")
include("search.jl")
include("PEPS.jl")
include("PEPS2.jl")
include("PEPS3.jl")
include("network_interface2.jl")
#include("network_interface3.jl")

end # module
