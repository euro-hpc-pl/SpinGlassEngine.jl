export
    PEPSNetwork,
    node_from_index,
    conditional_probability


# node's types to be changes
struct PEPSNetwork{T <: AbstractGeometry} <: AbstractGibbsNetwork{Node, Node}
    factor_graph::LabelledGraph{S, Node} where S
    network_graph  #TO BE REMOVEd
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    tensors_map::Dict
    gauges_data::Dict  # Should we store gauges explicitly here? pack into single structure?
    gauges_info

    function PEPSNetwork{T}(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation,
        initial_gauges::Symbol=:id
    ) where T <: AbstractGeometry

        vmap = vertex_map(transformation, m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        ng = network_graph(T, m, n) # TO BE REMOVED
        #ng = T.name.wrapper(m, n)

        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        tmap = tensor_map(T, nrows, ncols)

        gauges_data = Dict()
        gauges_info = gauges_list(T, nrows, ncols)

        net = new(factor_graph, ng, vmap, m, n, nrows, ncols,
                  tmap, gauges_data, gauges_info)

        initialize_gauges!(net)
        net
    end
end


function projectors(network::PEPSNetwork{T}, vertex::Node) where T <: Square
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
end


function projectors(network::PEPSNetwork{T}, vertex::Node) where T <: SquareStar
    i, j = vertex
    neighbours = (
                    ((i+1, j-1), (i, j-1), (i-1, j-1)), 
                    (i-1, j),
                    ((i+1, j+1), (i, j+1), (i-1, j+1)),
                    (i+1, j)
                )
    projector.(Ref(network), Ref(vertex), neighbours)
end


node_index(peps::AbstractGibbsNetwork, node::Node) = peps.ncols * (node[1] - 1) + node[2]

_mod_wo_zero(k, m) = k % m == 0 ? m : k % m

node_from_index(peps::AbstractGibbsNetwork, index::Int) =
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))


function boundary(peps::PEPSNetwork{T}, node::Node) where T <: Square
    i, j = node
    vcat(
        [
            ((i, k), (i+1, k)) for k ∈ 1:j-1
        ]...,
            ((i, j-1), (i, j)),
        [
            ((i-1, k), (i, k)) for k ∈ j:peps.ncols
        ]...
    )
end


function boundary(peps::PEPSNetwork{T}, node::Node) where T <: SquareStar
    i, j = node
    vcat(
        [
            [((i, k-1), (i+1, k), (i, k), (i+1, k-1)), ((i, k), (i+1, k))] for k ∈ 1:j-1
        ]...,
        ((i, j-1), (i+1, j)),
        ((i, j-1), (i, j)),
        ((i-1, j-1), (i, j)),
        ((i-1, j), (i, j)),
        [
            [((i-1, k-1), (i, k), (i-1, k), (i, k-1)), ((i-1, k), (i, k))] for k ∈ j+1:peps.ncols
        ]...
    )
end


function bond_energy(
    network::AbstractGibbsNetwork, 
    u::Node, 
    v::Node, 
    σ::Int
)
    fg_u, fg_v = network.vertex_map(u), network.vertex_map(v)
    if has_edge(network.factor_graph, fg_u, fg_v)
        pu, en, pv = get_prop.(Ref(network.factor_graph), Ref(fg_u), Ref(fg_v), (:pl, :en, :pr))
        energies = en[pu, pv[σ]]
    elseif has_edge(network.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(Ref(network.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr))
        energies = en[pv[σ], pu]
    else
        energies = zeros(length(local_energy(network, u)))
    end
    vec(energies)
end


function update_energy(network::PEPSNetwork{T}, σ::Vector{Int}) where T <: Square
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    local_energy(network, (i, j))
end


function update_energy(network::PEPSNetwork{T}, σ::Vector{Int}) where T <: SquareStar
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    bond_energy(network, (i, j), (i-1, j-1), local_state_for_node(network, σ, (i-1, j-1))) +
    bond_energy(network, (i, j), (i-1, j+1), local_state_for_node(network, σ, (i-1, j+1))) +
    local_energy(network, (i, j))
end
