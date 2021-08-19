using LabelledGraphs

export
    AbstractGibbsNetwork,
    network_graph,
    vertex_map,
    projectors,
    local_energy,
    interaction_energy,
    site_tensor,
    connecting_tensor,
    boundary_state,
    local_state_for_node,
    iteration_order,
    fuse_projectors,
    tensor,
    tensor_temp,
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

boundary_at_splitting_node(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T} = not_implmented("boundary_at_splitting_node")

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
        get_prop(fg, fg_w, fg_v, :pr)'
    elseif has_edge(fg, fg_v, fg_w)
        get_prop(fg, fg_v, fg_w, :pl)
    else
        loc_dim = fg_v ∈ vertices(fg) ? length(local_energy(network, v)) : 1 
        ones(loc_dim, 1)
    end
end


function projector(
    network::AbstractGibbsNetwork{S, T}, 
    v::S, 
    W::NTuple{N, S}
) where {S, T, N}
    proj = [projector(network, v, w) for w ∈ W]
    first(fuse_projectors(proj))
end


function fuse_projectors(projectors)
    fused, energy = rank_reveal(hcat(projectors...), :PE)
    i₀ = 1
    transitions = []
    for proj ∈ projectors
        iₑ = i₀ + size(proj, 2) - 1
        push!(transitions, energy[:, i₀:iₑ])
        i₀ = iₑ + 1
    end
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


@memoize function site_tensor(
    network::AbstractGibbsNetwork{S, T}, 
    v::S
) where {S, T}
    loc_exp = exp.(-network.β .* local_energy(network, v))
    projs = projectors(network, v)
    dim = zeros(Int, length(projs))

    @cast A[_, i] := loc_exp[i]
    for (j, pv) ∈ enumerate(projs)
        @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
        dim[j] = size(pv, 2)
    end
    Ã = reshape(A, dim..., :)
    dropdims(sum(Ã, dims=5), dims=5)
end


function tensor_temp(
    network::AbstractGibbsNetwork{S, T}, 
    v::Tuple{Int, Int}
) where {S, T}
    loc_exp = exp.(-network.β .* local_energy(network, v))
    projs = projectors(network, v)
    dim = zeros(Int, length(projs))

    @cast A[_, i] := loc_exp[i]
    for (j, pv) ∈ enumerate(projs)
        @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
        dim[j] = size(pv, 2)
    end
    reshape(A, dim..., :)
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Rational{Int}, Int},
    ::Val{:central_v}
) where {S, T}
    r, j = v
    i = floor(Int, r)
    h = connecting_tensor(network, (i, j), (i+1, j))
    @cast A[_, u, _, d] := h[u, d]
    A
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    w::Tuple{Int, Rational{Int}},
    ::Val{:central_h}
) where {S, T}
    i, r = w
    j = floor(Int, r)
    v = connecting_tensor(network, (i, j), (i, j+1))
    @cast A[l, _, r, _] := v[l, r]
    A
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Rational{Int}, Rational{Int}},
    ::Val{:central_d}
) where {S, T}
    r, s = v
    i = floor(Int, r)
    j = floor(Int, s)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1))
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j))
    @cast A[_, (u, ũ), _, (d, d̃)] := NW[u, d] * NE[ũ, d̃] 
    A
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Int, Rational{Int}},
    ::Val{:virtual}
) where {S, T}
    i, s = v
    j = floor(Int, s)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    h = connecting_tensor(network, (i, j), (i, j+1))

    @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]    
    @cast C[l, (ũ, u), r, (d̃, d)] |= B[l, r] * p_lt[l, u] * p_rb[r, d] * 
                                     p_rt[r, ũ] * p_lb[l, d̃]
    C
end


tensor(
    network::AbstractGibbsNetwork{S, T}, 
    v::S,
    ::Val{:site}
) where {S, T} = site_tensor(network, v)


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::R
) where {S, T, R}
    if v ∈ keys(network.tensor_spiecies)
        tensor(network, v, Val(network.tensor_spiecies[v]))
    else
        ones(1, 1, 1, 1)
    end
end


function tensor(
    network::AbstractGibbsNetwork{S, T}, 
    v::R,
    ::Val{:gauge_h}
) where {S, T, R}
    X = network.gauges[v]
    @cast A[_, u, _, d] := Diagonal(X)[u, d]
    A
end


@memoize function connecting_tensor(
    network::AbstractGibbsNetwork{S, T},
    v::S,
    w::S
) where {S, T}
    en = interaction_energy(network, v, w)
    exp.(-network.β .* (en .- minimum(en)))
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
    [findfirst(x -> x > 0, pv[i, :]) for i ∈ 1:size(pv)[1]][state]
end


function _boundary_index(
    network::AbstractGibbsNetwork{S, T}, 
    v::S, 
    w::S, 
    k::S, 
    l::S, 
    σ::Vector{Int}
) where {S, T}
    pv = projector(network, v, w)
    i = _boundary_index(network, v, w, σ)
    j = _boundary_index(network, k, l, σ)
    (j - 1) * size(pv, 2) + i
end


function boundary_state(
    network::AbstractGibbsNetwork{S, T},
    σ::Vector{Int},
    node::S
) where {S, T}
    [
        _boundary_index(network, x..., σ)
        for x ∈ boundary_at_splitting_node(network, node)
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
    network::AbstractGibbsNetwork,
    type::Symbol=:rand
) where T
    N = 6
    @assert type ∈ (:id, :rand)
    for i ∈ 1:network.nrows - 1, j ∈ 1:network.ncols
        a, b = size(interaction_energy(network, (i, j), (i + 1, j)))
        Y = type == :id ? ones(a) : rand(a) .+ 0.1
        push!(network.gauges, (i + 1//N, j) => Y, (i + 2//N, j) => 1 ./ Y)
        Z = type == :id ? ones(b) : rand(b) .+ 0.1
        push!(network.gauges, 
            (i + 4//N, j) => Z,
            (i + 5//N, j) => 1 ./ Z,
            (i + 1//N, j+1//2) => ones(1),
            (i + 2//N, j+1//2) => ones(1),
            (i + 4//N, j+1//2) => ones(1),
            (i + 5//N, j+1//2) => ones(1)
        )
    end
    for j ∈ 1:network.ncols
        push!(network.gauges, 
            (network.nrows + 1//N, j) => ones(1), (-1//N, j) => ones(1)
        )
    end
end
