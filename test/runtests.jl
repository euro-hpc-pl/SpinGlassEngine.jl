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
        #"search_chimera.jl", # OK
        #"search_smallest_chimera.jl", # OK except MPSAnnealing
        #"search_full_chimera.jl", #OK, all heuristics to be checked
        #"search_cross.jl", # OK, all heuristics to be checked
        #"search_pegasus.jl", # OK
        #"search_pegasus_nodiag.jl", # OK
        #"search_new_geometry_nodiag.jl", # NO
        #"chimera_overlap.jl", # OK
        #"gauges.jl", # OK NO
        #"gauges2.jl", # OK
        #"gauges3.jl",
        "memoization.jl",
        #"search_smallest_cross.jl", # OK
        #"small_instance.jl"
)

for my_test in my_tests
    include(my_test)
end
