using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs

disable_logging(LogLevel(1))

using Test
my_tests = []

push!(my_tests,
        "network_operations.jl",
        "branch_and_bound.jl",
        "network_interface.jl",
        "ising_MPS.jl",
        "search_MPS.jl",
        "search_chimera.jl",
        "search_cross.jl",
        "network_tensors.jl",
        "search_chimera2048.jl",
)

for my_test in my_tests
    include(my_test)
end
