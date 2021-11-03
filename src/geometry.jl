export
    IntOrRational,
    Node, PEPSNode,
    AbstractGeometry,
    AbstractTensorsLayout,
    tensor_map,
    gauges_list,
    Square,
    SquareStar,
    GaugesEnergy,
    EnergyGauges,
    EngGaugesEng


const IntOrRational = Union{Int, Rational{Int}}

abstract type AbstractGeometry end
abstract type AbstractConnectivity end
abstract type AbstractTensorsLayout end

struct SquareStar{T <: AbstractTensorsLayout} <: AbstractGeometry end
struct Square{T <: AbstractTensorsLayout} <: AbstractGeometry end 

const Chimera = Square
const Pegazus = SquareStar

# Somehow the names are not descriptive enough!
struct GaugesEnergy{T} <: AbstractTensorsLayout end
struct EnergyGauges{T} <: AbstractTensorsLayout end
struct EngGaugesEng{T} <: AbstractTensorsLayout end

const Node = NTuple{2, Int}

struct PEPSNode 
    i::IntOrRational
    j::IntOrRational
    
    PEPSNode(i, j) = new(denominator(i) == 1 ? numerator(i) : i,
                         denominator(j) == 1 ? numerator(j) : j)
end

Node(node::PEPSNode) = (node.i, node.j)

struct GaugeInfo
    positions::NTuple{2, PEPSNode}
    attached_tensor::PEPSNode
    attached_leg::Int
    type::Symbol
end


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


#-----------------------------
###    Square geometry     ###
#-----------------------------


function tensor_map(::Type{Square{T}}, 
    nrows::Int, 
    ncols::Int
) where T <: Union{GaugesEnergy, EnergyGauges}

    map = Dict{PEPSNode, Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => :site)
        if j < ncols push!(map, PEPSNode(i, j + 1//2) => :central_h) end
        if i < nrows push!(map, PEPSNode(i + 1//2, j) => :central_v) end
    end
    map
end


function gauges_list(::Type{Square{T}},
    nrows::Int,
    ncols::Int
) where T <: GaugesEnergy
    [
        GaugeInfo((PEPSNode(i + 1//6, j), PEPSNode(i + 2//6, j)), 
                   PEPSNode(i + 1//2, j), 1, :gauge_h) 
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end


function gauges_list(::Type{Square{T}},
    nrows::Int,
    ncols::Int
) where T <: EnergyGauges
    [
        GaugeInfo((PEPSNode(i + 4//6, j), PEPSNode(i + 5//6, j)), 
                   PEPSNode(i + 1//2, j), 2, :gauge_h) 
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end


function gauges_list(::Type{Square{T}},
    nrows::Int, 
    ncols::Int
) where T <: EngGaugesEng
    [
        GaugeInfo((PEPSNode(i + 2//5, j), PEPSNode(i + 3//5, j)), 
                   PEPSNode(i + 1//5, j), 2, :gauge_h) 
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end


#---------------------------------
###    SquareStar geometry     ###
#---------------------------------


function tensor_map(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T
    map = Dict{PEPSNode, Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => :site)
        push!(map, PEPSNode(i, j - 1//2) => :virtual)
        push!(map, PEPSNode(i + 1//2, j) => :central_v)
    end
    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(map, PEPSNode(i + 1//2, j + 1//2) => :central_d)
    end
    map
end


function gauges_list(::Type{SquareStar{T}},
    nrows::Int, 
    ncols::Int
) where T <: GaugesEnergy
    [
        GaugeInfo((PEPSNode(i + 1//6, j), PEPSNode(i + 2//6, j)), 
                   PEPSNode(i + 1//2, j), 1, :gauge_h) 
        for i ∈ 1:nrows-1 for j ∈ 1//2 : 1//2 : ncols
    ]
end


function gauges_list(::Type{SquareStar{T}},
    nrows::Int, 
    ncols::Int
) where T <: EnergyGauges
    [
        GaugeInfo((PEPSNode(i + 4//6, j), PEPSNode(i + 5//6, j)), 
                   PEPSNode(i + 1//2, j), 2, :gauge_h)
        for i ∈ 1:nrows-1 for j ∈ 1//2 : 1//2 : ncols
    ]
end


function gauges_list(::Type{SquareStar{T}},
    nrows::Int, 
    ncols::Int
) where T <: EngGaugesEng
    [
        GaugeInfo((PEPSNode(i + 2//5, j), PEPSNode(i + 3//5, j)), 
                   PEPSNode(i + 1//5, j), 2, :gauge_h) 
        for i ∈ 1:nrows-1 for j ∈ 1//2 : 1//2 : ncols
    ]
end
