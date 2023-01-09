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

using Test
my_tests = []

push!(my_tests,
# quick tests:
    #"operations.jl",
    #"branch_and_bound.jl",
    # "search_chimera_pathological.jl",
    # "search_chimera_smallest.jl",
    # "search_cross_square_star.jl",
    # "search_smallest_cross_square_star.jl",
    # "search_pegasus_square_star.jl",
    # "search_pegasus_nodiag_square_star.jl",
    # "search_square2_basic.jl",
    # "search_squarestar2_basic.jl",
    # "chimera_overlap_python.jl",

# to fix in Tensor
    #"search_chimera_pathological_gauge.jl",

# time consuming tests:
    # "search_chimera_full.jl",
    # "search_chimera_gauge.jl",
    # "gauges.jl"
)

# This is work in progress (may or may not be included in future versions)

push!(my_tests,
    # "experimental/zipper.jl",
    "experimental/squarestar2_pegasus.jl",
    # "experimental/squarestar2_zephyr.jl",
)


for my_test ∈ my_tests
    include(my_test)
end
