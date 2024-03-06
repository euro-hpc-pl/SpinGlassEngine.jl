module SpinGlassEngine

using Base: Tuple
using Base.Cartesian
using SpinGlassTensors
using SpinGlassNetworks
using CUDA
using TensorOperations
using TensorCast
using NNlib, NNlibCUDA
using MetaGraphs
using Memoization
using LinearAlgebra, MKL
using Graphs
using ProgressMeter
using Statistics
using DocStringExtensions
#using Infiltrator

include("operations.jl")
include("geometry.jl")
include("PEPS.jl")
include("contractor.jl")
include("square_single_node.jl")
include("square_cross_single_node.jl")
include("square_double_node.jl")
include("square_cross_double_node.jl")
include("tensors.jl")
include("droplets.jl")
include("search.jl")
include("util.jl")

end # module
