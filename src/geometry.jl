export Node, PEPSNode, AbstractGeometry, AbstractSparsity
export AbstractTensorsLayout, tensor_map, gauges_list, Dense, Sparse, Gauges
export Square, SquareStar, GaugesEnergy, EnergyGauges, EngGaugesEng, Pegasus

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
abstract type AbstractGeometry end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
abstract type AbstractSparsity end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
abstract type AbstractTensorsLayout end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct SquareStar{T <: AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct Square{T <: AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct Pegasus <: AbstractGeometry end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct Dense <: AbstractSparsity end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct Sparse <: AbstractSparsity end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct GaugesEnergy{T} <: AbstractTensorsLayout end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct EnergyGauges{T} <: AbstractTensorsLayout end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct EngGaugesEng{T} <: AbstractTensorsLayout end

"""
```julia
const Node = NTuple{N, Int} where N
```
"""
const Node = NTuple{N, Int} where N

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct PEPSNode
    i::Site
    j::Site

    function PEPSNode(i::Site, j::Site)
        new(denominator(i) == 1 ? numerator(i) : i, denominator(j) == 1 ? numerator(j) : j)
    end
end

"""
$(TYPEDSIGNATURES)

"""
Node(node::PEPSNode) = (node.i, node.j)

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct SuperPEPSNode
    i::Site
    j::Site
    k::Int

    function SuperPEPSNode(i::Site, j::Site, k::Int)
        new(denominator(i) == 1 ? numerator(i) : i, denominator(j) == 1 ? numerator(j) : j, k)
    end
end

"""
$(TYPEDSIGNATURES)

"""
Node(node::SuperPEPSNode) = (node.i, node.j, node.k)

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct GaugeInfo
    positions::NTuple{2, PEPSNode}
    attached_tensor::PEPSNode
    attached_leg::Int
    type::Symbol
end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct Gauges{T <: AbstractGeometry}
    data::Dict
    info::Vector{GaugeInfo}
    """
    ```julia
    function Gauges{T}(nrows::Int, ncols::Int) where T <: AbstractGeometry
    ```
    """
    function Gauges{T}(nrows::Int, ncols::Int) where T <: AbstractGeometry
        new(Dict(), gauges_list(T, nrows, ncols))
    end
end

"""
$(TYPEDSIGNATURES)

"""
function Square(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end

"""
$(TYPEDSIGNATURES)

"""
function SquareStar(m::Int, n::Int)
    lg = Square(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end

"""
$(TYPEDSIGNATURES)

"""
function Pegasus(m::Int, n::Int)
    labels = [(i, j, k) for j ∈ 1:n for i ∈ 1:m for k ∈ 1:2]
    lg = LabelledGraph(labels)
    for i ∈ 1:m, j ∈ 1:n add_edge!(lg, (i, j, 1), (i, j, 2)) end

    for i ∈ 1:m-1, j ∈ 1:n
        add_edge!(lg, (i, j, 1), (i+1, j, 1))
        add_edge!(lg, (i, j, 2), (i+1, j, 1))
    end

    for i ∈ 1:m, j ∈ 1:n-1
        add_edge!(lg, (i, j, 2), (i, j+1, 2))
        add_edge!(lg, (i, j, 1), (i, j+1, 2))
    end

    # for i ∈ 1:m-1, j ∈ 1:n-1  # diagonals
    #     add_edge!(lg, (i, j, 2), (i+1, j+1, 1))
    #     add_edge!(lg, (i, j, 1), (i+1, j+1, 2))
    # end
    lg
end

#-----------------------------
###    Square geometry     ###
#-----------------------------

"""
$(TYPEDSIGNATURES)

"""
site(::Type{Dense}) = :site

"""
$(TYPEDSIGNATURES)

"""
site(::Type{Sparse}) = :sparse_site

"""
$(TYPEDSIGNATURES)

"""
function tensor_map(
    ::Type{Square{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{GaugesEnergy, EnergyGauges}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site(S))
        if j < ncols push!(map, PEPSNode(i, j + 1//2) => :central_h) end
        if i < nrows push!(map, PEPSNode(i + 1//2, j) => :central_v) end
    end
    map
end

"""
$(TYPEDSIGNATURES)

"""
function tensor_map(
    ::Type{Square{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: EngGaugesEng, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site(S))
        if j < ncols push!(map, PEPSNode(i, j + 1//2) => :central_h) end
        if i < nrows
            push!(
                map,
                PEPSNode(i + 1//5, j) => :sqrt_up,
                PEPSNode(i + 4//5, j) => :sqrt_down
            )
         end
    end
    map
end

"""
$(TYPEDSIGNATURES)

"""
function gauges_list(::Type{Square{T}}, nrows::Int, ncols::Int) where T <: GaugesEnergy
    [
        GaugeInfo(
            (PEPSNode(i + 1//6, j), PEPSNode(i + 2//6, j)),
            PEPSNode(i + 1//2, j),
            1,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

"""
function gauges_list(::Type{Square{T}}, nrows::Int, ncols::Int) where T <: EnergyGauges
    [
        GaugeInfo(
            (PEPSNode(i + 4//6, j), PEPSNode(i + 5//6, j)),
            PEPSNode(i + 1//2, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

"""
function gauges_list(::Type{Square{T}}, nrows::Int, ncols::Int) where T <: EngGaugesEng
    [
        GaugeInfo(
            (PEPSNode(i + 2//5, j), PEPSNode(i + 3//5, j)),
            PEPSNode(i + 1//5, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

#---------------------------------
###    SquareStar geometry     ###
#---------------------------------

"""
$(TYPEDSIGNATURES)

"""
Virtual(::Type{Dense}) = :virtual

"""
$(TYPEDSIGNATURES)

"""
Virtual(::Type{Sparse}) = :sparse_virtual

"""
$(TYPEDSIGNATURES)

"""
function tensor_map(
    ::Type{SquareStar{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{EnergyGauges, GaugesEnergy}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site(S),
            PEPSNode(i, j - 1//2) => Virtual(S),
            PEPSNode(i + 1//2, j) => :central_v
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(map, PEPSNode(i + 1//2, j + 1//2) => :central_d)
    end
    map
end

"""
$(TYPEDSIGNATURES)

"""
function tensor_map(
    ::Type{SquareStar{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: EngGaugesEng, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site(S),
            PEPSNode(i, j - 1//2) => Virtual(S),
            PEPSNode(i + 1//5, j) => :sqrt_up,
            PEPSNode(i + 4//5, j) => :sqrt_down
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(
            map,
            PEPSNode(i + 1//5, j + 1//2) => :sqrt_up_d,
            PEPSNode(i + 4//5, j + 1//2) => :sqrt_down_d
        )
    end
    map
end

"""
$(TYPEDSIGNATURES)

"""
function gauges_list(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T <: GaugesEnergy
    [
        GaugeInfo(
            (PEPSNode(i + 1//6, j), PEPSNode(i + 2//6, j)),
            PEPSNode(i + 1//2, j),
            1,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

"""
function gauges_list(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T <: EnergyGauges
    [
        GaugeInfo(
            (PEPSNode(i + 4//6, j), PEPSNode(i + 5//6, j)),
            PEPSNode(i + 1//2, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

"""
function gauges_list(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T <: EngGaugesEng
    [
        GaugeInfo(
            (PEPSNode(i + 2//5, j), PEPSNode(i + 3//5, j)),
            PEPSNode(i + 1//5, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

#------------------------------
###    Pegasus geometry     ###
#------------------------------
# Geometry: 2 nodes -> 1 TN site. This will work for Chimera.

"""
$(TYPEDSIGNATURES)

"""
pegasus_site(::Type{Dense}) = :pegasus_site

"""
$(TYPEDSIGNATURES)

"""
pegasus_site(::Type{Sparse}) = :sparse_pegasus_site

"""
$(TYPEDSIGNATURES)

"""
function tensor_map(
    ::Type{Pegasus}, ::Type{S}, nrows::Int, ncols::Int
) where S <: AbstractSparsity
    map = Dict{PEPSNode, Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols push!(map, PEPSNode(i, j) => pegasus_site(S)) end
    map
end

"""
$(TYPEDSIGNATURES)

"""
function gauges_list(::Type{Pegasus}, nrows::Int, ncols::Int)
    [
        GaugeInfo(
            (PEPSNode(i + 1//3, j), PEPSNode(i + 2//3, j)),
            PEPSNode(i , j),
            4,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end
