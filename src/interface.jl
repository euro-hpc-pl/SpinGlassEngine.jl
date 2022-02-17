using LabelledGraphs

export AbstractGibbsNetwork, vertex_map, local_energy, interaction_energy, connecting_tensor
export normalize_probability, boundary_state, local_state_for_node
export fuse_projectors, initialize_gauges!, decode_state

# T: type of the vertex of network
# S: type of the vertex of underlying factor graph
"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
abstract type AbstractGibbsNetwork{S, T} end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct NotImplementedError{M} <: Exception
    m::M
    NotImplementedError(m::M) where {M} = new{M}(m)
end
not_implmented(name) = throw(NotImplementedError(name))

"""
$(TYPEDSIGNATURES)

"""
factor_graph(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.factor_graph

"""
$(TYPEDSIGNATURES)

"""
vertex_map(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.vertex_map

"""
$(TYPEDSIGNATURES)

"""
function boundary(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T}
    not_implmented("boundary_at_splitting_node")
end

"""
$(TYPEDSIGNATURES)

"""
function node_index(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T}
    not_implmented("node_index")
end

"""
$(TYPEDSIGNATURES)

"""
function update_energy(network::AbstractGibbsNetwork{S, T}, σ::Vector{Int}) where {S, T}
    not_implmented("update_energy")
end

"""
$(TYPEDSIGNATURES)

"""
function conditional_probability(
    network::AbstractGibbsNetwork{S, T}, v::Vector{Int}
) where {S, T}
    not_implemented("conditional_probability")
end

"""
$(TYPEDSIGNATURES)

"""
function projector(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    fg = factor_graph(network)
    fg_v, fg_w = network.vertex_map(v), network.vertex_map(w)
    if has_edge(fg, fg_w, fg_v)
        p = get_prop(fg, fg_w, fg_v, :pr)
    elseif has_edge(fg, fg_v, fg_w)
        p = get_prop(fg, fg_v, fg_w, :pl)
    else
        p = ones(Int, fg_v ∈ vertices(fg) ? cluster_size(network, v) : 1, 1)
    end
    vec(p)
end

"""
$(TYPEDSIGNATURES)

"""
function projector(
    network::AbstractGibbsNetwork{S, T}, v::S, vertices::NTuple{N, S}
) where {S, T, N}
    first(fuse_projectors(projector.(Ref(network), Ref(v), vertices)))
end

"""
$(TYPEDSIGNATURES)

"""
function fuse_projectors(projectors::Union{Vector{T}, NTuple{N, T}}) where {N, T}
    fused, transitions_matrix = rank_reveal(hcat(projectors...), :PE)
    transitions = collect(eachcol(transitions_matrix))
    fused, transitions
end

"""
$(TYPEDSIGNATURES)

"""
function spectrum(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    get_prop(factor_graph(network), vertex_map(network)(vertex), :spectrum)
end

"""
$(TYPEDSIGNATURES)

"""
function local_energy(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    spectrum(network, vertex).energies
end

"""
$(TYPEDSIGNATURES)

"""
function cluster_size(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    length(local_energy(network, vertex))
end

"""
$(TYPEDSIGNATURES)

"""
function interaction_energy(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    fg = factor_graph(network)
    fg_v, fg_w = network.vertex_map(v), network.vertex_map(w)
    if has_edge(fg, fg_w, fg_v)
        get_prop(fg, fg_w, fg_v, :en)'
    elseif has_edge(fg, fg_v, fg_w)
        get_prop(fg, fg_v, fg_w, :en)
    else
        zeros(1, 1)
    end
end

"""
$(TYPEDSIGNATURES)

"""
ones_like(x::Number) = one(typeof(x))

"""
$(TYPEDSIGNATURES)

"""
ones_like(x::AbstractArray) = ones(eltype(x), size(x))

"""
$(TYPEDSIGNATURES)

"""
function boundary_index(
    net::AbstractGibbsNetwork{S, T},
    nodes::Tuple{S, Union{S, NTuple{N, S}}},
    σ::Vector{Int}
) where {S, T, N}
    v, w = nodes
    state = local_state_for_node(net, σ, v)
    if net.vertex_map(v) ∉ vertices(net.factor_graph) return ones_like(state) end
    projector(net, v, w)[state]
end

"""
$(TYPEDSIGNATURES)

"""
function boundary_index(
    network::AbstractGibbsNetwork{S, T}, nodes::NTuple{4, S}, σ::Vector{Int}
) where {S, T}
    v, w, k, l = nodes
    pv = projector(network, v, w)
    i = boundary_index(network, (v, w), σ)
    j = boundary_index(network, (k, l), σ)
    (j - 1) * maximum(pv) + i
end

"""
$(TYPEDSIGNATURES)

"""
function boundary_state(
    network::AbstractGibbsNetwork{S, T}, σ::Vector{Int}, node::S
) where {S, T}
    boundary_index.(Ref(network), boundary(network, node), Ref(σ))
end

"""
$(TYPEDSIGNATURES)

"""
function local_state_for_node(
    network::AbstractGibbsNetwork{S, T}, σ::Vector{Int}, w::S
) where {S, T}
    k = node_index(network, w)
    0 < k <= length(σ) ? σ[k] : 1
end

"""
$(TYPEDSIGNATURES)

"""
function is_compatible(factor_graph::LabelledGraph, network_graph::LabelledGraph)
    all(has_edge(network_graph, src(edge), dst(edge)) for edge ∈ edges(factor_graph))
end

"""
$(TYPEDSIGNATURES)

"""
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

"""
$(TYPEDSIGNATURES)

"""
_normalize(probs::Vector{<:Real}) = probs ./ sum(probs)

"""
$(TYPEDSIGNATURES)

"""
function _equalize(probs::Vector{<:Real})
    mp = abs(minimum(probs))
    _normalize(replace(p -> p < mp ? mp : p, probs))
end

"""
$(TYPEDSIGNATURES)

"""
function normalize_probability(probs::Vector{<:Real})
    if minimum(probs) < 0 return _equalize(probs) end
    _normalize(probs)
end

"""
$(TYPEDSIGNATURES)

"""
function decode_state(
    peps::AbstractGibbsNetwork{S, T}, σ::Vector{Int}, map::Bool=false
) where {S, T}
    ver = iteration_order(peps)
    if map ver = peps.vertex_map.(ver) end
    Dict(ver[i] => σ[node_index(peps, ver[i])] for i ∈ 1:length(σ))
    #Dict(ver[i] => σ[i] for i ∈ 1:length(σ))
end
