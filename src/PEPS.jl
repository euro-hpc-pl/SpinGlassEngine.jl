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
       outer_projector

# T: type of the vertex of network
# S: type of the vertex of underlying factor graph
abstract type AbstractGibbsNetwork{S, T} end

"""
$(TYPEDSIGNATURES)
"""
mutable struct PEPSNetwork{
    T <: AbstractGeometry, S <: AbstractSparsity
} <: AbstractGibbsNetwork{Node, PEPSNode}
    factor_graph::LabelledGraph
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
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation,
        gauge_type::Symbol=:id
    ) where {T <: AbstractGeometry, S <: AbstractSparsity}
        lp = PoolOfProjectors{Int}()
        net = new(factor_graph, vertex_map(transformation, m, n), lp, m, n)
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

"""
$(TYPEDSIGNATURES)
"""
mod_wo_zero(k, m) = k % m == 0 ? m : k % m

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
function bond_energy(net::AbstractGibbsNetwork{T, S}, u::Node, v::Node, σ::Int) where {T, S}
    fg_u, fg_v = net.vertex_map(u), net.vertex_map(v)
    if has_edge(net.factor_graph, fg_u, fg_v)
        pu, en, pv = get_prop.(
                        Ref(net.factor_graph), Ref(fg_u), Ref(fg_v), (:pl, :en, :pr)
                    )
        @inbounds energies = en[pu, pv[σ]]
    elseif has_edge(net.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(
                        Ref(net.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr)
                    )
        @inbounds energies = en[pv[σ], pu]
    else
        energies = zeros(cluster_size(net, u))
    end
    vec(energies)
end

"""
$(TYPEDSIGNATURES)
"""
function projector(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    fg = network.factor_graph
    fg_v, fg_w = network.vertex_map(v), network.vertex_map(w)
    if has_edge(fg, fg_w, fg_v)
        p = get_prop(fg, fg_w, fg_v, :pr)
    elseif has_edge(fg, fg_v, fg_w)
        p = get_prop(fg, fg_v, fg_w, :pl)
    else
        p = ones(Int, fg_v ∈ vertices(fg) ? cluster_size(network, v) : 1)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function projector(
    net::AbstractGibbsNetwork{S, T}, v::S, vertices::NTuple{N, S}
) where {S, T, N}
    first(fuse_projectors(projector.(Ref(net), Ref(v), vertices)))
end

"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""
function spectrum(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    get_prop(network.factor_graph, network.vertex_map(vertex), :spectrum)
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
function SpinGlassNetworks.cluster_size(net::AbstractGibbsNetwork{S, T}, v::S) where {S, T}
    length(local_energy(net, v))
end

"""
$(TYPEDSIGNATURES)
"""
function interaction_energy(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    fg = network.factor_graph
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
    peps::AbstractGibbsNetwork{S, T}, σ::Vector{Int}, fg_order::Bool=false
) where {S, T}
    nodes = fg_order ? peps.vertex_map.(nodes_search_order_Mps(peps)) : vertices(peps.factor_graph)
    Dict(nodes[1:length(σ)] .=> σ)
end
