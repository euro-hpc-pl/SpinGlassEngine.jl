# PEPS.jl: This file provides all tools needed to build PEPS tensor network.

using LabelledGraphs

export
       AbstractGibbsNetwork,
       local_energy,
       interaction_energy,
       connecting_tensor,
       normalize_probability,
       fuse_projectors,
       initialize_gauges!,
       decode_state,
       PEPSNetwork,
       mod_wo_zero,
       bond_energy,
       outer_projector,
       bond_energy,
       projector,
       spectrum,
       interaction_energy,
       is_compatible

# T: type of the vertex of network
# S: type of the vertex of underlying factor graph
abstract type AbstractGibbsNetwork{S, T} end

"""
$(TYPEDSIGNATURES)
A mutable struct representing a Projected Entangled Pair States (PEPS) network.

# Fields
- `clustered_hamiltonian::LabelledGraph`: The clustered Hamiltonian associated with the network.
- `vertex_map::Function`: A function mapping vertex coordinates to a transformed lattice.
- `lp::PoolOfProjectors`: A pool of projectors used in the network.
- `m::Int`: The number of rows in the PEPS lattice.
- `n::Int`: The number of columns in the PEPS lattice.
- `nrows::Int`: The effective number of rows based on lattice transformations.
- `ncols::Int`: The effective number of columns based on lattice transformations.
- `tensors_map::Dict{PEPSNode, Symbol}`: A dictionary mapping PEPS nodes to tensor symbols.
- `gauges::Gauges{T}`: Gauges used for gauge fixing operations.
    
The `PEPSNetwork` struct represents a quantum network based on Projected Entangled Pair States (PEPS). It holds information about the clustered Hamiltonian, lattice transformations, and gauge fixing operations. PEPS networks are commonly used in quantum computing and quantum information theory to represent entangled states of qubits arranged on a lattice.
    
"""
mutable struct PEPSNetwork{
    T <: AbstractGeometry, S <: AbstractSparsity
} <: AbstractGibbsNetwork{Node, PEPSNode}
    clustered_hamiltonian::LabelledGraph
    vertex_map::Function
    lp::PoolOfProjectors
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    tensors_map::Dict{PEPSNode, Symbol}
    gauges::Gauges{T}

    function PEPSNetwork{T, S}(
        m::Int,
        n::Int,
        clustered_hamiltonian::LabelledGraph,
        transformation::LatticeTransformation,
        gauge_type::Symbol=:id
    ) where {T <: AbstractGeometry, S <: AbstractSparsity}
        lp = get_prop(clustered_hamiltonian, :pool_of_projectors)
        net = new(clustered_hamiltonian, vertex_map(transformation, m, n), lp, m, n)
        net.nrows, net.ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(net.clustered_hamiltonian, T.name.wrapper(m, n))
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        net.tensors_map = tensor_map(T, S, net.nrows, net.ncols)
        net.gauges = Gauges{T}(net.nrows, net.ncols)
        initialize_gauges!(net, gauge_type)
        net
    end
end

# """
# $(TYPEDSIGNATURES)
# """
mod_wo_zero(k, m) = k % m == 0 ? m : k % m

# """
# $(TYPEDSIGNATURES)
# """
ones_like(x::Number) = one(typeof(x))

# """
# $(TYPEDSIGNATURES)
# """
ones_like(x::AbstractArray) = ones(eltype(x), size(x))

# """
# $(TYPEDSIGNATURES)
# """
function bond_energy(net::AbstractGibbsNetwork{T, S}, u::Node, v::Node, σ::Int) where {T, S}
    cl_h_u, cl_h_v = net.vertex_map(u), net.vertex_map(v)
    energies = SpinGlassNetworks.bond_energy(net.clustered_hamiltonian, cl_h_u, cl_h_v, σ)
    vec(energies)
end

# """
# $(TYPEDSIGNATURES)
# """
function projector(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    cl_h = network.clustered_hamiltonian
    cl_h_v, cl_h_w = network.vertex_map(v), network.vertex_map(w)
    SpinGlassNetworks.projector(cl_h, cl_h_v, cl_h_w)
end

# """
# $(TYPEDSIGNATURES)
# """
function projector(
    net::AbstractGibbsNetwork{S, T}, v::S, vertices::NTuple{N, S}
) where {S, T, N}
    first(fuse_projectors(projector.(Ref(net), Ref(v), vertices)))
end

# """
# $(TYPEDSIGNATURES)
# """
function fuse_projectors(
    projectors::NTuple{N, K}
    #projectors::Union{Vector{S}, NTuple{N, S}}
    ) where {N, K}
    fused, transitions_matrix = rank_reveal(hcat(projectors...), :PE)
    # transitions = collect(eachcol(transitions_matrix))
    transitions = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))
    fused, transitions
end

function outer_projector(p1::Array{T, 1}, p2::Array{T, 1}) where T <: Number
    reshape(reshape(p1, :, 1) .+ maximum(p1) .* reshape(p2 .- 1, 1, :), :)
end

# """
# $(TYPEDSIGNATURES)
# """
function spectrum(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    get_prop(network.clustered_hamiltonian, network.vertex_map(vertex), :spectrum)
end

# """
# $(TYPEDSIGNATURES)
# """
function local_energy(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    spectrum(network, vertex).energies
end


# """
# $(TYPEDSIGNATURES)
# """
function SpinGlassNetworks.cluster_size(net::AbstractGibbsNetwork{S, T}, v::S) where {S, T}
    length(local_energy(net, v))
end

# """
# $(TYPEDSIGNATURES)
# """
function interaction_energy(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    cl_h = network.clustered_hamiltonian
    cl_h_v, cl_h_w = network.vertex_map(v), network.vertex_map(w)
    if has_edge(cl_h, cl_h_w, cl_h_v)
        get_prop(cl_h, cl_h_w, cl_h_v, :en)'
    elseif has_edge(cl_h, cl_h_v, cl_h_w)
        get_prop(cl_h, cl_h_v, cl_h_w, :en)
    else
        zeros(1, 1)
    end
end

# """
# $(TYPEDSIGNATURES)
# """
function is_compatible(clustered_hamiltonian::LabelledGraph, network_graph::LabelledGraph)
    all(has_edge(network_graph, src(edge), dst(edge)) for edge ∈ edges(clustered_hamiltonian))
end

# """
# $(TYPEDSIGNATURES)
# """
function initialize_gauges!(net::AbstractGibbsNetwork{S, T}, type::Symbol=:id) where {S, T}
    @assert type ∈ (:id, :rand)
    for gauge ∈ net.gauges.info
        n1, n2 = gauge.positions
        push!(net.tensors_map, n1 => gauge.type, n2 => gauge.type)
        d = size(net, gauge.attached_tensor)[gauge.attached_leg]
        X = type == :id ? ones(d) : rand(d) .+ 0.42
        push!(net.gauges.data, n1 => X, n2 => 1 ./ X)
    end
end

# """
# $(TYPEDSIGNATURES)
# """
_normalize(probs::Vector{<:Real}) = probs ./ sum(probs)
function _equalize(probs::Vector{<:Real})
    mp = abs(minimum(probs))
    _normalize(replace(p -> p < mp ? mp : p, probs))
end

# """
# $(TYPEDSIGNATURES)
# """
function normalize_probability(probs::Vector{<:Real})
    if minimum(probs) < 0 return _equalize(probs) end
    _normalize(probs)
end

# """
# $(TYPEDSIGNATURES)
# """
function decode_state(
    peps::AbstractGibbsNetwork{S, T}, σ::Vector{Int}, cl_h_order::Bool=false
) where {S, T}
    nodes = cl_h_order ? peps.vertex_map.(nodes_search_order_Mps(peps)) : vertices(peps.clustered_hamiltonian)
    Dict(nodes[1:length(σ)] .=> σ)
end
