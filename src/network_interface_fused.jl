using LabelledGraphs

export fuse_projectors, build_tensor, generate_boundary_states


function fuse_projectors(projectors) #::NTuple{N, Matrix{T}}) where {N, T}
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


function generate_boundary_state(
    network::FusedNetwork,
    v::S,
    w::S, 
    σ::Vector{Int}
) where S
    state = local_state_for_node(network, σ, v)
    if v ∉ vertices(network.network_graph) return ones_like(state) end
    pv = projector(network, v, w)
    [findfirst(x -> x > 0, pv[i, :]) for i ∈ 1:size(pv)[1]][state]
end


function generate_boundary_state(
    network::FusedNetwork, 
    v::S, 
    w::S, 
    k::S, 
    σ::Vector{Int}
) where S
    state = local_state_for_node(network, σ, v)
    if v ∉ vertices(network.network_graph) return ones_like(state) end

    pv = projector(network, v, w)
    pk = projector(network, v, k)

    P, _ = fuse_projectors([pv, pk])
    [findfirst(x -> x > 0, P[i, :]) for i ∈ 1:size(P)[1]][state]
end


function generate_boundary_state(
    network::FusedNetwork, 
    v::S, 
    w::S, 
    k::S, 
    l::S, 
    σ::Vector{Int}
) where S

    state_v = local_state_for_node(network, σ, v)
    if v ∉ vertices(network.network_graph) return ones_like(state_v) end

    state_k = local_state_for_node(network, σ, k)
    pv, pk = projector(network, v, w), projector(network, k, l)

    ind_v = [findfirst(x -> x > 0, pv[i, :]) for i ∈ 1:size(pv)[1]][state_v]
    ind_k = [findfirst(x -> x > 0, pk[i, :]) for i ∈ 1:size(pk)[1]][state_k]

    (ind_k - 1) * size(pv, 2) + ind_v
end
