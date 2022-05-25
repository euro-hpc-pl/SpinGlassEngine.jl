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
        #"ising_MPS.jl",
        #"search_MPS.jl",
        #"search_chimera_pathological.jl", # OK
        #"search_chimera_smallest.jl", # OK except MPSAnnealing
        #"search_chimera_full.jl", #OK, all heuristics to be checked
        #"search_cross_square_star.jl", # OK, all heuristics to be checked
        #"search_smallest_cross_square_star.jl", # OK
        #"search_pegasus_square_star.jl", # OK
        #"search_pegasus_nodiag_square_star.jl", # OK
        #"future_tests/chimera_overlap_python.jl", # OK
        #"future_tests/cross_square_star_prob.jl",
        #"future_tests/search_new_geometry_nodiag.jl", # NO
        #"future_tests/pegasus_nondiag_geometry.jl", # NO Pegasus not defined
        "future_tests/gauges.jl", # OK
        "future_tests/gauges2.jl", # OK
        "future_tests/gauges3.jl",
        "future_tests/memoization.jl",
)

for my_test in my_tests
    include(my_test)
end
