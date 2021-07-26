using LabelledGraphs

export fuse_projectors, build_tensor_with_fusing, generate_boundary_states_with_fusing


function generate_boundary_state_with_fusing(network::NNNNetwork, v::S, w::S, σ::Vector{Int}) where {S, T}
    v_lin = node_index_with_fusing(network, v)
    state =  0 < v_lin <= length(σ) ? σ[v_lin] : 1
    if v ∉ vertices(network.network_graph) return ones_like(state) end
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
    (ind_k - 1) * size(pv, 2) + ind_v
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
        generate_boundary_state_with_fusing.(Ref(network), Ref(x)..., σ)
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