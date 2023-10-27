export
    exact_marginal_probability,
    exact_conditional_probability

    """
$(TYPEDSIGNATURES)
"""
@memoize function exact_spectrum(clustered_hamiltonian::LabelledGraph{S, T}) where {S, T}  
    # TODO: Not going to work without PoolOfProjectors
    ver = vertices(clustered_hamiltonian)
    rank = cluster_size.(Ref(clustered_hamiltonian), ver)
    states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
    energy.(Ref(clustered_hamiltonian), states), states
end

"""
$(TYPEDSIGNATURES)
"""
function exact_marginal_probability(ctr::MpsContractor{T}, σ::Vector{Int}) where T 
    # TODO: Not going to work without PoolOfProjectors
    target_state = decode_state(ctr.peps, σ, true)
    energies, states = exact_spectrum(ctr.peps.clustered_hamiltonian)
    prob = exp.(-ctr.betas[end] .* energies)
    prob ./= sum(prob)
    sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
end

"""
$(TYPEDSIGNATURES)
"""
function exact_conditional_probability(ctr::MpsContractor{T}, σ::Vector{Int}) where T 
    # TODO: Not going to work without PoolOfProjectors
    local_basis = collect(1:cluster_size(ctr.peps, ctr.current_node))
    probs = exact_marginal_probability.(Ref(ctr), branch_state(local_basis, σ))
    probs ./= sum(probs)
end
