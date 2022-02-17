module SpinGlassEngine

using MKL
using SpinGlassTensors
using SpinGlassNetworks
using TensorOperations
using TensorCast
using MetaGraphs
using Memoize
using LinearAlgebra
using LightGraphs
using ProgressMeter

using DocStringExtensions
#using Infiltrator


# to be remove
function SpinGlassNetworks.local_basis(ψ::AbstractMPS, i::Int)
    SpinGlassNetworks.local_basis(physical_dim(ψ, i))
end

# # to be remove
# function LinearAlgebra.dot(ψ::AbstractMPS, state::Union{Vector, NTuple})
#     C = I
#     for (M, σ) ∈ zip(ψ, state)
#         i = idx(σ)
#         C = M[:, i, :]' * (C * M[:, i, :])
#     end
#     tr(C)
# end

include("geometry.jl")
include("operations.jl")
include("interface.jl")
include("MPS_search.jl")
include("PEPS.jl")
include("contractor.jl")
include("tensors.jl")
include("search.jl")

end # module
