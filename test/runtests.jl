using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
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
# quick tests:
     #"operations.jl",
     #"branch_and_bound.jl",
     #"search_chimera_pathological.jl",
     #"search_chimera_smallest.jl",
     #"search_cross_square_star.jl",
     #"search_smallest_cross_square_star.jl",
     #"search_cross_square_star.jl",
     #"search_smallest_cross_square_star.jl",
     #"search_pegasus_square_star.jl",
     #"search_pegasus_nodiag_square_star.jl", # 4x4 instance is missing

# time consuming tests:
    #"search_chimera_full.jl",
)

# This is work in progress (may or may not be included in future versions)

push!(my_tests,
    # "experimental/chimera_overlap_python.jl", # OK
    # "experimental/cross_square_star_prob.jl",
    #"experimental/pegasus_diag_basic.jl",
    #"experimental/pegasus_nondiag_basic.jl",
    # "experimental/pegasus_nondiag_heavy.jl",
    "experimental/square2_reading.jl",
    # "experimental/gauges.jl",
    # "experimental/memoization.jl",
    # "experimental/experiments_with_sparse_chimera.jl",
    # "experimental/zephyr_lattice.jl",
    # "experimental/zephyr_lattice_z1.jl",
    #"experimental/zephyr_lattice_square.jl"
)


for my_test ∈ my_tests
    include(my_test)
end
