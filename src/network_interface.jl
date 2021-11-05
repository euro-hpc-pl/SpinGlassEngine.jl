# This file should be clean !

using LabelledGraphs

export
    AbstractGibbsNetwork,
    vertex_map,
    local_energy,
    interaction_energy,
    connecting_tensor,
    normalize_probability,
    boundary_state,
    local_state_for_node,
    iteration_order,
    fuse_projectors,
    initialize_gauges!


# T: type of the vertex of network
# S: type of the vertex of underlying factor graph
abstract type AbstractGibbsNetwork{S, T} end


struct NotImplementedError{M} <: Exception
    m::M
    NotImplementedError(m::M) where {M} = new{M}(m)
end 

not_implmented(name) = throw(NotImplementedError(name)) 

 
factor_graph(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.factor_graph

vertex_map(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.vertex_map

boundary(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T} = not_implmented("boundary_at_splitting_node")

node_index(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T} = not_implmented("node_index")

update_energy(network::AbstractGibbsNetwork{S, T}, σ::Vector{Int}) where {S, T} = not_implmented("update_energy")

conditional_probability(network::AbstractGibbsNetwork{S, T}, v::Vector{Int}) where {S, T} = not_implemented("conditional_probability")

iteration_order(peps::AbstractGibbsNetwork{T, S}) where {S, T} = [(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols]


function projector(  # ta funkcja ma zwracac 1d projectors
    network::AbstractGibbsNetwork{S, T}, 
    v::S, 
    w::S
) where {S, T}
    fg = factor_graph(network)
    vmap = vertex_map(network)
    fg_v, fg_w = vmap(v), vmap(w)
    
    if has_edge(fg, fg_w, fg_v)
        get_prop(fg, fg_w, fg_v, :pr)
    elseif has_edge(fg, fg_v, fg_w)
        get_prop(fg, fg_v, fg_w, :pl)
    else
        loc_dim = fg_v ∈ vertices(fg) ? length(local_energy(network, v)) : 1 
        floor.(Int, ones(loc_dim, 1))
    end
end


function projector(
    network::AbstractGibbsNetwork{S, T}, 
    v::S, 
    W::NTuple{N, S}
) where {S, T, N}
    first(fuse_projectors(
        [projector(network, v, w) for w ∈ W]
    ))
end

function fuse_projectors(projectors::Union{Vector{T}, NTuple{N, T}}) where {N, T}
    fused, transitions_matrix = rank_reveal(hcat(projectors...), :PE)
    transitions = collect(eachcol(transitions_matrix))
    fused, transitions
end

function spectrum(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    get_prop(factor_graph(network), vertex_map(network)(vertex), :spectrum)
end


function local_energy(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    spectrum(network, vertex).energies
end


function interaction_energy(
    network::AbstractGibbsNetwork{S, T}, 
    v::S, 
    w::S
) where {S, T}
    fg = factor_graph(network)
    vmap = vertex_map(network)
    fg_v, fg_w = vmap(v), vmap(w)
    if has_edge(fg, fg_w, fg_v)
        get_prop(fg, fg_w, fg_v, :en)'
    elseif has_edge(fg, fg_v, fg_w)
        get_prop(fg, fg_v, fg_w, :en)
    else
        zeros(1, 1)
    end
end


ones_like(x::Number) = one(typeof(x))

ones_like(x::Array) = ones(eltype(x), size(x))


function _boundary_index(
    network::AbstractGibbsNetwork{S, T}, 
    v::S, 
    w::Union{S, NTuple{N, S}},
    σ::Vector{Int}
) where {S, T, N}
    state = local_state_for_node(network, σ, v)
    if v ∉ vertices(network.factor_graph) return ones_like(state) end
    pv = projector(network, v, w) 
    #[findfirst(x -> x > 0, pv[i, :]) for i ∈ 1:size(pv)[1]][state]
    pv[state]
end


function _boundary_index(
    network::AbstractGibbsNetwork{S, T}, 
    v::S, w::S, k::S, l::S, 
    σ::Vector{Int}
) where {S, T}
    pv = projector(network, v, w)
    i = _boundary_index(network, v, w, σ)
    j = _boundary_index(network, k, l, σ)
    (j - 1) * maximum(pv) + i
end


function boundary_state(
    network::AbstractGibbsNetwork{S, T},
    σ::Vector{Int},
    node::S
) where {S, T} 
     [
        _boundary_index(network, x..., σ)
        for x ∈ boundary(network, node)
    ] 
end


function local_state_for_node(
    network::AbstractGibbsNetwork{S, T},
    σ::Vector{Int},
    w::S
) where {S, T}
    k = node_index(network, w)
    0 < k <= length(σ) ? σ[k] : 1
end


function is_compatible(
    factor_graph::LabelledGraph,
    network_graph::LabelledGraph,
) 
    all(
        has_edge(network_graph, src(edge), dst(edge))
        for edge ∈ edges(factor_graph)
    )
end


function initialize_gauges!(
    network::AbstractGibbsNetwork{S, T},
    type::Symbol=:rand
) where {S, T}
    @assert type ∈ (:id, :rand)
    for gauge ∈ network.gauges_info
        (n1, n2) = gauge.positions
        push!(network.tensors_map, n1 => gauge.type, n2 => gauge.type)
        d = tensor_size(network, gauge.attached_tensor)[gauge.attached_leg]
        X = type == :id ? ones(d) : rand(d) .+ 0.42
        push!(network.gauges_data, n1 => X, n2 => 1 ./ X)
    end
end


function normalize_probability(values::Vector{R}) where {R <: Number}
    minp = minimum(values)
    if minp < 0
        amp = abs(minp)
        for (i, p) ∈ enumerate(values)
            p < amp ? values[i] = amp : values[i]
        end
    end
    values / sum(values)
end
