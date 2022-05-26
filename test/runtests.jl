using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

disable_logging(LogLevel(1))

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
    for (A, σ) ∈ zip(ψ, state) C *= A[:, idx(σ), :] end
    tr(C)
end

function SpinGlassEngine.tensor(ψ::MPS)
    dims = rank(ψ)
    Θ = Array{eltype(ψ)}(undef, dims)
    for σ ∈ all_states(dims) Θ[idx.(σ)...] = tensor(ψ, σ) end
    Θ
end

using Test
my_tests = []


push!(my_tests,
    #"operations.jl",
    #"branch_and_bound.jl",
    #"search_chimera_pathological.jl",
    #"search_chimera_smallest.jl", # problem with MPSAnnealing
    #"search_cross_square_star.jl",  # final test fails
    #"search_smallest_cross_square_star.jl",
    "search_chimera_full.jl",   #  ground is incorrect
    #"search_cross_square_star.jl",  # final test fails
    #"search_smallest_cross_square_star.jl",
    #"search_pegasus_square_star.jl"
    #"search_pegasus_nodiag_square_star.jl"
)

# This is work in progress (may or may not be included in furure versions)
#=
push!(my_tests,
    #"future_tests/chimera_overlap_python.jl", # OK
    #"future_tests/cross_square_star_prob.jl",
    #"future_tests/search_new_geometry_nodiag.jl", # NO
    #"future_tests/pegasus_nondiag_geometry.jl", # NO
    #"future_tests/gauges.jl", # O
    #"future_tests/memoization.jl",
)
=#

for my_test in my_tests
    include(my_test)
end
