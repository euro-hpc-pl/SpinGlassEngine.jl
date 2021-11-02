export
    IntOrRational,
    RNode,
    AbstractGeometry,
    AbstractTensorsLayout,
    network_graph,
    tensor_map,
    gauges_list,
    Square,
    SquareStar,
    GaugesEnergy,
    EnergyGauges,
    EngGaugesEng


const IntOrRational = Union{Int, Rational{Int}}
const RNode = NTuple{2, IntOrRational}

abstract type AbstractGeometry end
abstract type AbstractConnectivity end
abstract type AbstractTensorsLayout end

# Should we use Chimera and Pegazus instead of Square and SquareStar?
struct SquareStar{T <: AbstractTensorsLayout} <: AbstractGeometry end
struct Square{T <: AbstractTensorsLayout} <: AbstractGeometry end 

# Do aliases make sense?
const Chimera = Square
const Pegazus = SquareStar

# These are OK in general, but somehow the names are not descriptive enough!
struct GaugesEnergy{T} <: AbstractTensorsLayout end
struct EnergyGauges{T} <: AbstractTensorsLayout end
struct EngGaugesEng{T} <: AbstractTensorsLayout end

# add types
struct GaugeInfo
    positions
    attached_tensor
    attached_leg
    type
end

#=
struct Node
    i::IntOrRational
    j::IntOrRational 
    
    function Node(i::IntOrRational, j::IntOrRational)
        Node(denominator(i) == 1 ? numerator(i) : i,
             denominator(j) == 1 ? numerator(j) : j)
    end
end
=#

#=
# This is different idea to be explored 
function Square(m::Int, n::Int) 
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


function SquareStar(m::Int, n::Int)
    lg = Square(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end
=#


##############################
###    Square geometry     ###
##############################


function network_graph(::Type{Square{T}}, m::Int, n::Int) where T
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


function tensor_map(::Type{Square{T}}, 
    nrows::Int, 
    ncols::Int
) where T <: Union{GaugesEnergy, EnergyGauges}

    map = Dict{RNode, Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, (i, j) => :site)
        if j < ncols push!(map, (i, j + 1//2) => :central_h) end
        if i < nrows push!(map, (i + 1//2, j) => :central_v) end
    end
    map
end


function gauges_list(::Type{Square{T}},
    nrows::Int,
    ncols::Int
) where T <: GaugesEnergy
    [GaugeInfo(((i + 1//6, j), (i + 2//6, j)), (i + 1//2, j), 1, :gauge_h) for i ∈ 1:nrows-1 for j ∈ 1:ncols]
end


function gauges_list(::Type{Square{T}},
    nrows::Int,
    ncols::Int
) where T <: EnergyGauges
    [GaugeInfo(((i + 4//6, j), (i + 5//6, j)), (i + 1//2, j), 2, :gauge_h) for i ∈ 1:nrows-1 for j ∈ 1:ncols]
end


function gauges_list(::Type{Square{T}},
    nrows::Int, 
    ncols::Int
) where T <: EngGaugesEng
    [GaugeInfo(((i + 2//5, j), (i + 3//5, j)), (i + 1//5, j), 2, :gauge_h) for i ∈ 1:nrows-1 for j ∈ 1:ncols]
end


##################################
###    SquareStar geometry     ###
##################################


function network_graph(::Type{SquareStar{T}}, m::Int, n::Int) where T
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    lg = LabelledGraph(labels, grid((m, n)))
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end


function tensor_map(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T
    map = Dict{RNode, Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, (i, j) => :site)
        push!(map, (i, j - 1//2) => :virtual)
        push!(map, (i + 1//2, j) => :central_v)
    end
    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(map, (i + 1//2, j + 1//2) => :central_d)
    end
    map
end

# in dispatrching there is a problem with 1//1 vs 1

function gauges_list(::Type{SquareStar{T}},
    nrows::Int, 
    ncols::Int
) where T <: GaugesEnergy
    gl = [GaugeInfo(((i + 1//6, j), (i + 2//6, j)), (i + 1//2, j), 1, :gauge_h) for i ∈ 1:nrows-1 for j ∈ 1 : ncols]
    for i ∈ 1:nrows-1, j ∈ 1//2:ncols
        push!(gl, GaugeInfo(((i + 1//6, j), (i + 2//6, j)), (i + 1//2, j), 1, :gauge_h))
    end
    gl
end


function gauges_list(::Type{SquareStar{T}},
    nrows::Int, 
    ncols::Int
) where T <: EnergyGauges
    gl = [GaugeInfo(((i + 4//6, j), (i + 5//6, j)), (i + 1//2, j), 2, :gauge_h) for i ∈ 1:nrows-1 for j ∈ 1 : ncols]
    for i ∈ 1:nrows-1, j ∈ 1//2:ncols
        push!(gl, GaugeInfo(((i + 4//6, j), (i + 5//6, j)), (i + 1//2, j), 2, :gauge_h))
    end
    gl
end


function gauges_list(::Type{SquareStar{T}},
    nrows::Int, 
    ncols::Int
) where T <: EngGaugesEng
    gl = [GaugeInfo(((i + 2//5, j), (i + 3//5, j)), (i + 1//5, j), 2, :gauge_h) for i ∈ 1:nrows-1 for j ∈ 1 : ncols]
    for i ∈ 1:nrows-1, j ∈ 1//2:ncols
        push!(gl, GaugeInfo(((i + 2//5, j), (i + 3//5, j)), (i + 1//5, j), 2, :gauge_h))
    end
    gl
end
