export PEPSNetwork, node_from_index

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
        transformation::LatticeTransformation
    ) where {T <: AbstractGeometry, S <: AbstractSparsity}
        net = new(factor_graph, vertex_map(transformation, m, n), m, n)
        net.nrows, net.ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        # if !is_compatible(net.factor_graph, T.name.wrapper(m, n))
        #     throw(ArgumentError("Factor graph not compatible with given network."))
        # end

        net.tensors_map = tensor_map(T, S, net.nrows, net.ncols)
        net.gauges = Gauges{T}(net.nrows, net.ncols)
        initialize_gauges!(net)
        net
    end
end

function projectors(network::PEPSNetwork{T, S}, vertex::Node) where {T <: Square, S}
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
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


function projectors(network::PEPSNetwork{T, S}, vertex::Node) where {T <: Pegasus, S}
    i, j = vertex
    (
        projector(network, (i, j-1, 2), ((i, j, 1), (i, j, 2))),
        projector(network, (i-1, j, 1), ((i, j, 1), (i, j, 2))),
        projector(network, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))),
        projector(network, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))
    )
end


function node_index(peps::AbstractGibbsNetwork{T, S}, node::Node) where {T, S}
    peps.ncols * (node[begin] - 1) + node[end]
end

_mod_wo_zero(k, m) = k % m == 0 ? m : k % m
function node_from_index(peps::PEPSNetwork{T, S}, index::Int) where {T <: Square, S}
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))
end

function node_from_index(peps::PEPSNetwork{T, S}, index::Int) where {T <: SquareStar, S}
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))
end


function node_from_index(peps::PEPSNetwork{T, S}, index::Int) where {T <: Pegasus, S}
    ((index-1) ÷ (2 * peps.ncols) + 1, _mod_wo_zero(index, 2* peps.ncols), _mod_wo_zero(index, 2))
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
        [((i-1, k, 1), ((i, k, 1), (i, k, 2))) for k ∈ j:peps.ncols]...
    )
end


function bond_energy(
    network::AbstractGibbsNetwork{T, S}, u::Node, v::Node, σ::Int
) where {T, S}
    fg_u, fg_v = network.vertex_map(u), network.vertex_map(v)

    if has_edge(network.factor_graph, fg_u, fg_v)
        pu, en, pv = get_prop.(
                        Ref(network.factor_graph), Ref(fg_u), Ref(fg_v), (:pl, :en, :pr)
                    )
        energies = @view en[pu, pv[σ]]
    elseif has_edge(network.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(
                        Ref(network.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr)
                    )
        energies = @view en[pv[σ], pu]
    else
        energies = zeros(length(local_energy(network, u)))
    end
    vec(energies)
end

function update_energy(network::PEPSNetwork{T, S}, σ::Vector{Int}) where {T <: Square, S}
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    local_energy(network, (i, j))
end

function update_energy(
    network::PEPSNetwork{T, S}, σ::Vector{Int}
) where {T <: SquareStar, S}
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    bond_energy(network, (i, j), (i-1, j-1), local_state_for_node(network, σ, (i-1, j-1))) +
    bond_energy(network, (i, j), (i-1, j+1), local_state_for_node(network, σ, (i-1, j+1))) +
    local_energy(network, (i, j))
end


function update_energy(network::PEPSNetwork{T, S}, σ::Vector{Int}) where {T <: Pegasus, S}
    i, j, k = node_from_index(network, length(σ)+1)
    en = bond_energy(network, (i, j, k), (i, j-1, 2), local_state_for_node(network, σ, (i, j-1, 2))) +
    bond_energy(network, (i, j, k), (i-1, j, 1), local_state_for_node(network, σ, (i-1, j, 1))) +
    local_energy(network, (i, j, k))
    if k == 2
        en += bond_energy(network, (i, j, 2), (i, j, 1), local_state_for_node(network, σ, (i, j, 1)))
    end
    en
end