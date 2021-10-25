using LabelledGraphs

export
    AbstractGibbsNetwork,
    network_graph,
    vertex_map,
    local_energy,
    interaction_energy,
    connecting_tensor,
    normalize_probability,
    boundary_state,
    local_state_for_node,
    iteration_order,
    fuse_projectors,
    update_gauges!

# S: type of the vertex of network
# T: type of the vertex of underlying factor graph
abstract type AbstractGibbsNetwork{S, T} end


struct NotImplementedError{M} <: Exception
    m::M
    NotImplementedError(m::M) where {M} = new{M}(m)
end 

not_implmented(name) = throw(NotImplementedError(name)) 

 
factor_graph(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.factor_graph

network_graph(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.network_graph

vertex_map(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.vertex_map

boundary(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T} = not_implmented("boundary_at_splitting_node")

node_index(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T} = not_implmented("node_index")

update_energy(network::AbstractGibbsNetwork{S, T}, σ::Vector{Int}) where {S, T} = not_implmented("update_energy")

conditional_probability(network::AbstractGibbsNetwork{S, T}, v::Vector{Int}) where {S, T} = not_implemented("conditional_probability")

iteration_order(peps::AbstractGibbsNetwork) = [(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols]


function projector(
    network::AbstractGibbsNetwork{S, T}, 
    v::S, 
    w::S
) where {S, T}
    fg = factor_graph(network)
    vmap = vertex_map(network)
    fg_v, fg_w = vmap(v), vmap(w)
    
    if has_edge(fg, fg_w, fg_v)
        P = get_prop(fg, fg_w, fg_v, :pr)#'
        #decode_projector!(P, :EP)'  # to out
    elseif has_edge(fg, fg_v, fg_w)
        P = get_prop(fg, fg_v, fg_w, :pl)
        #decode_projector!(P, :PE) # to out
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
    if v ∉ vertices(network.network_graph) return ones_like(state) end
    pv = projector(network, v, w) 
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
    factor_graph::LabelledGraph{T, NTuple{2, Int}}, 
    network_graph::LabelledGraph{S, NTuple{2, Int}}
) where {T, S}
    all(
        has_edge(network_graph, src(edge), dst(edge))
        for edge ∈ edges(factor_graph)
    )
end


function update_gauges!(
    network::AbstractGibbsNetwork{S, T},
    type::Symbol=:rand
) where {S, T}
    @assert type ∈ (:id, :rand)
    for i ∈ 1:network.nrows-1, k ∈ 1//2 : 1//2 : network.ncols
        j = denominator(k) == 1 ? numerator(k) : k
        u, d = tensor_size(network, (i + 1//2, j))
        Y = type == :id ? ones(u) : rand(u) .+ 0.42
        push!(network.gauges, (i + 1//6, j) => Y, (i + 2//6, j) => 1 ./ Y)
        Z = type == :id ? ones(d) : rand(d) .+ 0.42
        push!(network.gauges, (i + 4//6, j) => Z, (i + 5//6, j) => 1 ./ Z)
    end
end


function normalize_probability(values::Vector{R}) where {R<:Number}
    minp = minimum(values)
    if minp < 0
        amp = abs(minp)
        for (i, p) ∈ enumerate(values)
            p < amp ? values[i] = amp : values[i]
        end
    end
    values / sum(values)
end
