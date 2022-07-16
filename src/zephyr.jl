export ZephyrSquare

"""
$(TYPEDSIGNATURES)
"""
struct ZephyrSquare <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)
"""
function ZephyrSquare(m::Int, n::Int)
    labels = [(i, j, k) for i ∈ 1:2*n for j ∈ 1:2*m for k ∈ 1:2]  # change for bigger zephyr
    lg = LabelledGraph(labels)
    for i ∈ 1:2*n, j ∈ j_function(i, n) add_edge!(lg, (i, j, 1), (i, j, 2)) end

    # horizontals
    for i ∈ 1:2*n, j ∈ j_function(i, n)[begin:end-1]
        add_edge!(lg, (i, j, 1), (i, j+1, 2))
        add_edge!(lg, (i, j, 2), (i, j+1, 1))
    end

    #vertical
    for i ∈ 1:2*n - 1 , j ∈ j_function(i, n)
        add_edge!(lg, (i, j, 1), (i+1, j, 2))
        add_edge!(lg, (i, j, 2), (i+1, j, 1))

    end

    # diagonals
    for i ∈ 1:2*n - 1, j ∈ j_function(i, n)[begin:end-1]
        add_edge!(lg, (i, j, 2), (i+1, j+1, 2))
        add_edge!(lg, (i+1, j, 1), (i, j+1, 1))
    end

    for i ∈ 1:n-1, j ∈ j_function(i, n)[begin]
        add_edge!(lg, (i, j, 1), (i+1, j-1, 1))
    end

    for i ∈ 1:n-1, j ∈ j_function(i, n)[end]
        add_edge!(lg, (i, j, 2), (i+1, j+1, 2))
    end

    for i ∈ n+1:2*n-1, j ∈ j_function(i, n)[begin]
       add_edge!(lg, (i, j, 2), (i+1, j+1, 2))
    end

    for i ∈ n+1:2*n-1, j ∈ j_function(i, n)[end]
        add_edge!(lg, (i, j, 1), (i+1, j-1, 1))
    end

    lg
end


"""
$(TYPEDSIGNATURES)

Geometry: 2 nodes -> 1 TN site. This will work for Chimera.
"""
zephyr_square_site(::Type{Dense}) = :zephyr_square_site

"""
$(TYPEDSIGNATURES)
"""
zephyr_square_site(::Type{Sparse}) = :sparse_zephyr_square_site

"""
$(TYPEDSIGNATURES)
"""
function tensor_map(
    ::Type{ZephyrSquare}, ::Type{S}, nrows::Int, ncols::Int
) where S <: AbstractSparsity
    map = Dict{PEPSNode, Symbol}()
    for i ∈ 1:2*nrows, j ∈ j_function(i, nrows)
        push!(map, PEPSNode(i, j) => zephyr_square_site(S)) 
    end
    map
end

"""
$(TYPEDSIGNATURES)
"""
function gauges_list(::Type{T}, nrows::Int, ncols::Int) where T <: ZephyrSquare
[
    GaugeInfo(
        (PEPSNode(i + 1//3, j), PEPSNode(i + 2//3, j)),
        PEPSNode(i , j),
        4,
        :gauge_h
    )
    for i ∈ 1:2*nrows-1 for j ∈ j_function(i, nrows)
]
end

"""
$(TYPEDSIGNATURES)
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: ZephyrSquare
    MpoLayers(
        Dict(i => (-1//3, 0, 1//3) for i ∈ 1:2*ncols),
        Dict(i => (1//3,) for i ∈ 1:2*ncols),
        Dict(i => (0,) for i ∈ 1:2*ncols)
    )
end

"""
$(TYPEDSIGNATURES)
"""
function projectors_site_tensor(net::PEPSNetwork{T, S}, vertex::Node) where {T <: ZephyrSquare, S}
    i, j = vertex
    (
        projector(net, (i, j-1, 2), ((i, j, 1), (i, j, 2))),
        projector(net, (i-1, j, 1), ((i, j, 1), (i, j, 2))),
        projector(net, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))),
        projector(net, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))
    )
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::PEPSNetwork{ZephyrSquare, T}, node::PEPSNode, ::Val{:zephyr_square_site}
) where T <: AbstractSparsity
    maximum.(projectors_site_tensor(network, Node(node)))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::PEPSNetwork{ZephyrSquare, T}, node::PEPSNode, ::Val{:sparse_zephyr_square_site}
) where T <: AbstractSparsity
    maximum.(projectors_site_tensor(network, Node(node)))
end