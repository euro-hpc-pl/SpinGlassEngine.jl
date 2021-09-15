using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs

# disable_logging(LogLevel(1))

using Test

function proj(state, dims::Union{Vector, NTuple})
    P = Matrix{Float64}[]
    for (σ, r) ∈ zip(state, dims)
        v = zeros(r)
        v[idx(σ)...] = 1.
        push!(P, v * v')
    end
    P
end

function SpinGlassEngine.tensor(ψ::AbstractMPS, state::State)
    C = I
    for (A, σ) ∈ zip(ψ, state)
        C *= A[:, idx(σ), :]
    end
    tr(C)
end

function SpinGlassEngine.tensor(ψ::MPS)
    dims = rank(ψ)
    Θ = Array{eltype(ψ)}(undef, dims)

    for σ ∈ all_states(dims)
        Θ[idx.(σ)...] = tensor(ψ, σ)
    end
    Θ
end

using Test
my_tests = []

push!(my_tests,
#=
        "network_operations.jl",
        "branch_and_bound.jl",
        "network_interface.jl",
        "ising_MPS.jl",
        "search_MPS.jl",
        "search_chimera.jl",
        "search_cross.jl",
        "network_tensors.jl",
        "search_full_chimera.jl",
=#
        "compressions.jl"
)

for my_test in my_tests
    include(my_test)
end
