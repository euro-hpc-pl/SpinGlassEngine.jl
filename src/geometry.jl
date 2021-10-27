export 
    AbstractGeometry, 
    AbstractTensorsLayout,
    network_graph,
    Square,
    Star

abstract type AbstractGeometry end
abstract type AbstractConnectivity end
abstract type AbstractTensorsLayout end

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
    map = Dict()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, (i, j) => :site)
        if j < ncols push!(map, (i, j + 1//2) => :central_h) end
        if i < nrows push!(map, (i + 1//2, j) => :central_v) end
    end
    map
end


function tensor_map(::Type{Square{Star}}, nrows::Int, ncols::Int)
    map = Dict()
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


function initialize_gauges!(::Type{Square}, map::Dict, nrows::Int, ncols::Int)
    for i ∈ 1:nrows-1, j ∈ 1:ncols
        push!(map, (i + 4//6, j) => :gauge_h)
        push!(map, (i + 5//6, j) => :gauge_h)
    end
    for i ∈ 1:nrows-1, j ∈ 1:ncols
        push!(map, (i + 1//6, j) => :gauge_h)
        push!(map, (i + 2//6, j) => :gauge_h)
    end
end


function initialize_gauges!(::Type{Square{Star}}, map::Dict, nrows::Int, ncols::Int)
    for i ∈ 1 : nrows - 1, j ∈ 1//2 : 1//2 : ncols
        jj = denominator(j) == 1 ? numerator(j) : j
        push!(map, (i + 4//6, jj) => :gauge_h)
        push!(map, (i + 5//6, jj) => :gauge_h)
    end
end

