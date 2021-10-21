module SpinGlassEngine

using Base: Tuple
using SpinGlassTensors, SpinGlassNetworks
using TensorOperations, TensorCast
using MetaGraphs
using Memoize
using LinearAlgebra
using LightGraphs
using ProgressMeter

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
include("geometry.jl")
include("s_network_interface.jl")
include("MPS_search.jl")
include("s_PEPS.jl")
include("s_PEPS_fused.jl")
include("contractor.jl")
include("s_network_tensors.jl")
include("search.jl")

end # module
