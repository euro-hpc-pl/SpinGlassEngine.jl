module SpinGlassEngine

using Base: Tuple
using SpinGlassTensors
using SpinGlassNetworks
using CUDA
using TensorOperations
using TensorCast
using MetaGraphs
using Memoization
using LinearAlgebra, MKL
using LightGraphs
using ProgressMeter
using Statistics
using DocStringExtensions
#using Infiltrator

# to be remove
function SpinGlassNetworks.local_basis(ψ::AbstractMPS, i::Int)
    SpinGlassNetworks.local_basis(physical_dim(ψ, i))
end

# to be remove
function LinearAlgebra.dot(ψ::AbstractMPS, state::Union{Vector, NTuple})
    C = I
    for (M, σ) ∈ zip(ψ, state)
        i = idx(σ)
        C = M[:, i, :]' * (C * M[:, i, :])
    end
    tr(C)
end

include("operations.jl")
include("geometry.jl")
include("PEPS.jl")
include("contractor.jl")
include("square.jl")
include("square_star.jl")
include("pegasus_nd.jl")
#include("pegasus.jl") # <-- conflicts with pegasus_nd
include("zephyr.jl")
include("tensors.jl")
include("search.jl")

end # module
