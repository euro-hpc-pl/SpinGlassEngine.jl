using LabelledGraphs

export fuse_projectors, build_tensor_with_fusing, generate_boundary_states_with_fusing


#function fuse_projectors(projectors)
#    fused, energy = rank_reveal(hcat(projectors...), :PE)
#    i₀ = 1
#    transitions = []
#    for proj ∈ projectors
#        iₑ = i₀ + size(proj, 2) - 1
#        push!(transitions, energy[:, i₀:iₑ])
#        i₀ = iₑ + 1
#    end
#    fused, transitions
#end


# This has to be unified with build_tensor
#@memoize function build_tensor_with_fusing(network::AbstractGibbsNetwork{S, T}, v::S) where {S, T}
#    loc_exp = exp.(-network.β .* local_energy(network, v))

    #(pl, pt, pr, pb), trl, trr
#    projs, trl, trr = projectors_with_fusing(network, v) # only difference in comparison to build_tensor
#    dim = zeros(Int, length(projs))
#    @cast A[_, i] := loc_exp[i]
    #v = build_tensor(peps, (i-1, j), (i, j)) ###
    #@tensor A[l, u, r, d] := v[u, ũ] * A[l, ũ, r, d] ###

#    for (j, pv) ∈ enumerate(projs)
#        @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
#        dim[j] = size(pv, 2)
#    end
#    reshape(A, dim..., :), trl, trr 
#end


function generate_boundary_state_with_fusing(network::NNNNetwork, v::S, w::S, state) where {S, T}
    if v ∉ vertices(network.network_graph) return ones_like(state) end
    loc_dim = length(local_energy(network, v))
    pv = projector(network, v, w)
    [findfirst(x -> x > 0, pv[i, :]) for i ∈ 1:size(pv)[1]][state]
end


function generate_boundary_state_with_fusing(network::NNNNetwork, v::S, w::S, k::S, σ::Vector{Int}) where {S, T}
    v_lin = node_index_with_fusing(network, v)
    state =  0 < v_lin <= length(σ) ? σ[v_lin] : 1
    if v ∉ vertices(network.network_graph) return ones_like(state) end
    pv = projector(network, v, w)
    pk = projector(network, v, k)
    P, _ = fuse_projectors([pv, pk])
    [findfirst(x -> x > 0, P[i, :]) for i ∈ 1:size(P)[1]][state]
end

function generate_boundary_state_with_fusing(network::NNNNetwork, v::S, w::S, k::S, l::S, σ::Vector{Int}) where {S, T}
    v_lin = node_index_with_fusing(network, v)
    state_v = 0 < v_lin <= length(σ) ? σ[v_lin] : 1
    if v ∉ vertices(network.network_graph) return ones_like(state_v) end
    k_lin = node_index_with_fusing(network, k)
    state_k = 0 < k_lin <= length(σ) ? σ[k_lin] : 1
    pv = projector(network, v, w)
    ind_v = [findfirst(x -> x > 0, pv[i, :]) for i ∈ 1:size(pv)[1]][state_v]
    pk = projector(network, k, l)
    ind_k = [findfirst(x -> x > 0, pk[i, :]) for i ∈ 1:size(pk)[1]][state_k]
    println(ind_v * size(pk, 2) + ind_k)
    ind_v * size(pk, 2) + ind_k
end


function generate_boundary_states_with_fusing(
    network::NNNNetwork,
    σ::Vector{Int},
    node::S
) where {S, T}
    [
        generate_boundary_state_with_fusing(network, x..., σ)
        for x ∈ boundary_at_splitting_node(network, node)
    ]
end


function generate_boundary_states_with_fusing(
    network::NNNNetwork,
    σ::Vector{Vector{Int}},
    node::S
) where {S, T}
    [
        generate_boundary_state_with_fusing(network, x..., σ)
        for x ∈ boundary_at_splitting_node(network, node)
    ]
end

function local_state_for_node(
    network::NNNNetwork,
    σ::Vector{Int},
    w::S
) where {S, T}
    k = node_index_with_fusing(network, w)
    0 < k <= length(σ) ? σ[k] : 1
end