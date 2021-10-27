export 
    IntOrRational, 
    RNode,
    AbstractTensor,
    Site,
    Virtual,
    Reduced,
    Central,
    Gauge,
    AbstractGeometry, 
    AbstractTensorsLayout,
    network_graph,
    Square,
    Star

const IntOrRational = Union{Int, Rational{Int}}
const RNode = NTuple{2, IntOrRational}

abstract type AbstractTensor end
abstract type AbstractGeometry end
abstract type AbstractConnectivity end
abstract type AbstractTensorsLayout end

struct Site <: AbstractTensor end
struct Virtual <: AbstractTensor end
struct Reduced <: AbstractTensor end
struct Central{T} <: AbstractTensor end
struct Gauge{T} <: AbstractTensor end

struct Star <: AbstractConnectivity end
struct Square{S <: AbstractConnectivity, 
              T <: AbstractTensorsLayout} <: AbstractGeometry 
end 


function network_graph(::Type{Square}, m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


function network_graph(::Type{Square{Star}}, m::Int, n::Int) 
    lg = network_graph(Square, m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end


function tensor_map(::Type{Square}, nrows::Int, ncols::Int) 
    map = Dict{RNode, AbstractTensor}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, (i, j) => Site)
        if j < ncols push!(map, (i, j + 1//2) => Central{Horizontal}) end
        if i < nrows push!(map, (i + 1//2, j) => Central{Vertical}) end
    end
    map
end


function tensor_map(::Type{Square{Star}}, nrows::Int, ncols::Int)
    map = Dict{RNode, AbstractTensor}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, (i, j) => Site)
        push!(map, (i, j - 1//2) => Virtual)
        push!(map, (i + 1//2, j) => Central{Vertical})
    end
    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(map, (i + 1//2, j + 1//2) => Central{Diagonal})
    end
    map
end


function initialize_gauges!(::Type{Square}, 
    map::Dict{RNode, T},
    nrows::Int, 
    ncols::Int
) where T <: AbstractTensor
    for i ∈ 1:nrows-1, j ∈ 1:ncols
        push!(map, (i + 4//6, j) => Gauge)
        push!(map, (i + 5//6, j) => Gauge)
    end
    for i ∈ 1:nrows-1, j ∈ 1:ncols
        push!(map, (i + 1//6, j) => Gauge)
        push!(map, (i + 2//6, j) => Gauge)
    end
end


function initialize_gauges!(::Type{Square{Star}}, 
    map::Dict{RNode, T}, 
    nrows::Int, 
    ncols::Int
) where T <: AbstractTensor
    for i ∈ 1 : nrows - 1, j ∈ 1//2 : 1//2 : ncols
        jj = denominator(j) == 1 ? numerator(j) : j
        push!(map, (i + 4//6, jj) => :Gauge)
        push!(map, (i + 5//6, jj) => :Gauge)
    end
end

