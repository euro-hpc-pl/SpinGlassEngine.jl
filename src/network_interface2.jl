using LabelledGraphs

export fuse_projectors, build_tensor_with_fusing, generate_boundary_states_with_fusing, generate_boundary_state_with_fusing_2


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


# This has to be unified with build_tensor
@memoize function build_tensor_with_fusing(network::AbstractGibbsNetwork{S, T}, v::S) where {S, T}
    loc_exp = exp.(-network.β .* local_energy(network, v))

    #(pl, pt, pr, pb), trl, trr
    projs, trl, trr = projectors_with_fusing(network, v) # only difference in comparison to build_tensor
    dim = zeros(Int, length(projs))
    @cast A[_, i] := loc_exp[i]

    #v = build_tensor(peps, (i-1, j), (i, j)) ###
    #@tensor A[l, u, r, d] := v[u, ũ] * A[l, ũ, r, d] ###

    for (j, pv) ∈ enumerate(projs)
        @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
        dim[j] = size(pv, 2)
    end

    reshape(A, dim..., :), trl, trr 
end

function generate_boundary_state_with_fusing(network::PegasusNetwork, v::S, w::S, state) where {S, T}
    if v ∉ vertices(network.network_graph) return ones_like(state) end
    loc_dim = length(local_energy(network, v))
    pv = projector(network, v, w)
    [findfirst(x -> x > 0, pv[i, :]) for i ∈ 1:size(pv)[1]][state]
end

function generate_boundary_state_with_fusing_2(network::PegasusNetwork, v::S, w::S, k::S, state) where {S, T}
    if v ∉ vertices(network.network_graph) return ones_like(state) end
    loc_dim = length(local_energy(network, v))
    pv = projector(network, v, w)
    pk = projector(network, v, k)
    P, _ = fuse_projectors([pv, pk])
    [findfirst(x -> x > 0, P[i, :]) for i ∈ 1:size(P)[1]][state]
end


function generate_boundary_states_with_fusing(
    network::PegasusNetwork,
    σ::Vector{Int},
    node::S
) where {S, T}
B = []
for x ∈ boundary_at_splitting_node(network, node)
    if length(x) == 2
        v, w = x
        push(B, generate_boundary_state_with_fusing(network, v, w, local_state_for_node(network, σ, v)))
    elseif length(x) == 3
        v, w, k = x
        push(B, generate_boundary_state_with_fusing_2(network, v, w, k, local_state_for_node(network, σ, v)))
    end
end
B
    #[
        #generate_boundary_state_with_fusing(network, v, w, local_state_for_node(network, σ, v))
        #for (v, w) ∈ boundary_at_splitting_node(network, node)
    #]
end


function generate_boundary_states_with_fusing(
    network::PegasusNetwork,
    σ::Vector{Vector{Int}},
    node::S
) where {S, T}
B = []
for x ∈ boundary_at_splitting_node(network, node)
    if length(x) == 2
        v, w = x
        push(B, generate_boundary_state_with_fusing(network, v, w, local_state_for_node.(Ref(network), σ, Ref(v))))
    elseif length(x) == 3
        v, w, k = x
        push(B, generate_boundary_state_with_fusing_2(network, v, w, k, local_state_for_node.(Ref(network), σ, Ref(v))))
    end
end
B
    #[
        #generate_boundary_state_with_fusing(network, v, w, local_state_for_node(network, σ, v))
        #for (v, w) ∈ boundary_at_splitting_node(network, node)
    #]
end



#function generate_boundary_states_with_fusing(
#    network::PegasusNetwork,
#    σ::Vector{Int},
#    node::S
#) where {S, T}
#    [
#        generate_boundary_state_with_fusing(network, v, w, local_state_for_node(network, σ, v))
#        for (v, w) ∈ boundary_at_splitting_node(network, node)
#    ]
#end


#function generate_boundary_states_with_fusing(
#    network::PegasusNetwork,
#    σ::Vector{Vector{Int}},
#    node::S
#) where {S, T}
#    [
#        generate_boundary_state_with_fusing(network, v, w, local_state_for_node.(Ref(network), σ, Ref(v)))
#        for (v, w) ∈ boundary_at_splitting_node(network, node)
#    ]
#end

function local_state_for_node(
    network::PegasusNetwork,
    σ::Vector{Int},
    w::S
) where {S, T}
    k = node_index_with_fusing(network, w)
    0 < k <= length(σ) ? σ[k] : 1
end