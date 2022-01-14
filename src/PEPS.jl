export PEPSNetwork, node_from_index, iteration_order

mutable struct PEPSNetwork{
    T <: AbstractGeometry, S <: AbstractSparsity
} <: AbstractGibbsNetwork{Node, PEPSNode}
    factor_graph::LabelledGraph
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    tensors_map::Dict
    gauges::Gauges{T}

    function PEPSNetwork{T, S}(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation,
        gauge_type::Symbol=:id
    ) where {T <: AbstractGeometry, S <: AbstractSparsity}
        net = new(factor_graph, vertex_map(transformation, m, n), m, n)
        net.nrows, net.ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(net.factor_graph, T.name.wrapper(m, n))
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        net.tensors_map = tensor_map(T, S, net.nrows, net.ncols)
        net.gauges = Gauges{T}(net.nrows, net.ncols)
        initialize_gauges!(net, gauge_type)
        net
    end
end

function projectors(network::PEPSNetwork{T, S}, vertex::Node) where {T <: Square, S}
    i, j = vertex
    projector.(Ref(network), Ref(vertex), ((i, j-1), (i-1, j), (i, j+1), (i+1, j)))
end

function projectors(network::PEPSNetwork{T, S}, vertex::Node) where {T <: SquareStar, S}
    i, j = vertex
    nbrs = (
        ((i+1, j-1), (i, j-1), (i-1, j-1)),
        (i-1, j),
        ((i+1, j+1), (i, j+1), (i-1, j+1)),
        (i+1, j)
    )
    projector.(Ref(network), Ref(vertex), nbrs)
end

function projectors(net::PEPSNetwork{T, S}, vertex::Node) where {T <: Pegasus, S}
    i, j = vertex
    (
        projector(net, (i, j-1, 2), ((i, j, 1), (i, j, 2))),
        projector(net, (i-1, j, 1), ((i, j, 1), (i, j, 2))),
        projector(net, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))),
        projector(net, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))
    )
end

function node_index(peps::AbstractGibbsNetwork{T, S}, node::Node) where {T, S}
    peps.ncols * (node[begin] - 1) + node[end]
end

mod_wo_zero(k, m) = k % m == 0 ? m : k % m
function node_from_index(peps::PEPSNetwork{T, S}, index::Int) where {T <: Square, S}
    ((index - 1) ÷ peps.ncols + 1, mod_wo_zero(index, peps.ncols))
end

function node_from_index(peps::PEPSNetwork{T, S}, index::Int) where {T <: SquareStar, S}
    ((index - 1) ÷ peps.ncols + 1, mod_wo_zero(index, peps.ncols))
end

function node_from_index(peps::PEPSNetwork{T, S}, idx::Int) where {T <: Pegasus, S}
    (
        (idx - 1) ÷ (2 * peps.ncols) + 1,
        mod_wo_zero((idx - 1) ÷ 2 + 1, peps.ncols),
        mod_wo_zero(idx, 2)
    )
end

function iteration_order(peps::PEPSNetwork{T, S}) where {T <: Union{Square,SquareStar}, S}
    [(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols]
end

function iteration_order(peps::PEPSNetwork{T, S}) where {T <: Pegasus, S}
    [(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2]
end

function boundary(peps::PEPSNetwork{T, S}, node::Node) where {T <: Square, S}
    i, j = node
    vcat(
        [((i, k), (i+1, k)) for k ∈ 1:j-1]...,
        ((i, j-1), (i, j)),
        [((i-1, k), (i, k)) for k ∈ j:peps.ncols]...
    )
end

function boundary(peps::PEPSNetwork{T, S}, node::Node) where {T <: SquareStar, S}
    i, j = node
    vcat(
        [
            [((i, k-1), (i+1, k), (i, k), (i+1, k-1)), ((i, k), (i+1, k))]
            for k ∈ 1:(j-1)
        ]...,
        ((i, j-1), (i+1, j)),
        ((i, j-1), (i, j)),
        ((i-1, j-1), (i, j)),
        ((i-1, j), (i, j)),
        [
            [((i-1, k-1), (i, k), (i-1, k), (i, k-1)), ((i-1, k), (i, k))]
            for k ∈ (j+1):peps.ncols
        ]...
    )
end

function boundary(peps::PEPSNetwork{T, S}, node::Node) where {T <: Pegasus, S}
    i, j = node
    vcat(
        [((i, k, 1), ((i+1, k, 1), (i+1, k, 2))) for k ∈ 1:j-1]...,
        ((i, (j-1), 2), ((i, j, 1), (i, j, 2))),
        [((i-1, k, 1), ((i, k, 1), (i, k, 2))) for k ∈ j:peps.ncols]...,
        ((i, j, 1), (i, j, 2))
    )
end

function bond_energy(net::AbstractGibbsNetwork{T, S}, u::Node, v::Node, σ::Int) where {T, S}
    fg_u, fg_v = net.vertex_map(u), net.vertex_map(v)
    if has_edge(net.factor_graph, fg_u, fg_v)
        pu, en, pv = get_prop.(
                        Ref(net.factor_graph), Ref(fg_u), Ref(fg_v), (:pl, :en, :pr)
                    )
        energies = en[pu, pv[σ]]
    elseif has_edge(net.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(
                        Ref(net.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr)
                    )
        energies = en[pv[σ], pu]
    else
        energies = zeros(cluster_size(net, u))
    end
    vec(energies)
end

@inline node_neighbors(::Type{Square{T}}, (i, j)) where {T} = ((i, j-1), (i-1, j))
@inline function node_neighbors(::Type{SquareStar{T}}, (i, j)) where T
    ((i, j-1), (i-1, j), (i-1, j-1), (i-1, j+1))
end
@inline node_neighbors(::Type{Pegasus}, (i, j, k)) = ((i, j-1, 2), (i-1, j, 1))

function update_energy(net::PEPSNetwork{T, S}, σ::Vector{Int}) where {T, S}
    @infiltrate
    u = node_from_index(net, length(σ)+1)
    en = local_energy(net, u)
    for v ∈ node_neighbors(T, u)
        en += bond_energy(net, u, v, local_state_for_node(net, σ, v))
    end
    @infiltrate
    if T != Pegasus return en end
    i, j, k = u
    if k != 2 return en end
    en += bond_energy(net, u, (i, j, 1), local_state_for_node(net, σ, (i, j, 1)))
    @infiltrate
    en
end
