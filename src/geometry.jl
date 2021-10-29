export 
    IntOrRational, 
    RNode,
    AbstractGeometry, 
    AbstractTensorsLayout,
    network_graph,
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


# This is different idea to be explored 
#=
function (::Square)(m::Int, n::Int) 
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


function (::SquareStar)(m::Int, n::Int)
    lg = Square(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end
=#

function network_graph(::Type{Square{T}}, m::Int, n::Int) where T
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


function network_graph(::Type{SquareStar{T}}, m::Int, n::Int) where T
    lg = network_graph(Square{T}, m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
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

# This should be probably dispatched differently
# Also, do we really need gauge_pairs?
function initialize_gauges!(::Type{Square{T}}, 
    map::Dict{RNode, Symbol}, 
    nrows::Int,
    ncols::Int
) where T <: GaugesEnergy

    gauge_pairs = []
    for i ∈ 1:nrows-1, j ∈ 1:ncols
        p, q = (i + 1//6, j), (i + 2//6, j)
        push!(map, p => :gauge_h)
        push!(map, q => :gauge_h)
        push!(gauge_pairs, (p, q))
    end
    gauge_pairs
end


function initialize_gauges!(::Type{Square{T}}, 
    map::Dict{RNode, Symbol}, 
    nrows::Int, 
    ncols::Int
) where T <: EnergyGauges

    gauge_pairs = []
    for i ∈ 1:nrows-1, j ∈ 1:ncols
        p, q = (i + 4//6, j), (i + 5//6, j)
        push!(map, p => :gauge_h)
        push!(map, q => :gauge_h)
        push!(gauge_pairs, (p, q))
    end
    gauge_pairs
end


function initialize_gauges!(::Type{Square{T}},
    map::Dict{RNode, Symbol}, 
    nrows::Int, 
    ncols::Int
) where T <: EngGaugesEng

    gauge_pairs = []
    for i ∈ 1:nrows-1, j ∈ 1:ncols
        p, q = (i + 2//5, j), (i + 3//5, j)
        push!(map, p => :gauge_h)
        push!(map, q => :gauge_h)
        push!(gauge_pairs, (p, q))
    end
    gauge_pairs
end


function initialize_gauges!(::Type{SquareStar{T}}, 
    map::Dict{RNode, Symbol}, 
    nrows::Int, 
    ncols::Int
) where T <: EnergyGauges
    for i ∈ 1 : nrows - 1, j ∈ 1//2 : 1//2 : ncols
        jj = denominator(j) == 1 ? numerator(j) : j
        push!(map, (i + 4//6, jj) => :gauge_h)
        push!(map, (i + 5//6, jj) => :gauge_h)
    end
end


function initialize_gauges!(::Type{SquareStar{T}}, 
    map::Dict{RNode, Symbol}, 
    nrows::Int, 
    ncols::Int
) where T <: GaugesEnergy
    for i ∈ 1 : nrows - 1, j ∈ 1//2 : 1//2 : ncols
        jj = denominator(j) == 1 ? numerator(j) : j
        push!(map, (i + 1//6, jj) => :gauge_h)
        push!(map, (i + 2//6, jj) => :gauge_h)
    end
end


function initialize_gauges!(::Type{SquareStar{T}}, 
    map::Dict{RNode, Symbol}, 
    nrows::Int, 
    ncols::Int
) where T <: EngGaugesEng
    for i ∈ 1 : nrows - 1, j ∈ 1//2 : 1//2 : ncols
        jj = denominator(j) == 1 ? numerator(j) : j
        push!(map, (i + 2//5, jj) => :gauge_h)
        push!(map, (i + 3//5, jj) => :gauge_h)
    end
end
