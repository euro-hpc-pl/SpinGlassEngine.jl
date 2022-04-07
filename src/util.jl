export
    exact_marginal_probability,
    exact_conditional_probability

    """
$(TYPEDSIGNATURES)
"""
@memoize function exact_spectrum(factor_graph::LabelledGraph{S, T}) where {S, T}
    ver = vertices(factor_graph)
    rank = cluster_size.(Ref(factor_graph), ver)
    states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
    energy.(Ref(factor_graph), states), states
end

"""
$(TYPEDSIGNATURES)
"""
function exact_marginal_probability(ctr::MpsContractor{T}, σ::Vector{Int}) where T
    target_state = decode_state(ctr.peps, σ, true)
    energies, states = exact_spectrum(ctr.peps.factor_graph)
    prob = exp.(-ctr.betas[end] .* energies)
    prob ./= sum(prob)
    sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
end

"""
$(TYPEDSIGNATURES)
"""
function exact_conditional_probability(ctr::MpsContractor{T}, σ::Vector{Int}) where T
    local_basis = collect(1:cluster_size(ctr.peps, ctr.current_node))
    probs = exact_marginal_probability.(Ref(ctr), branch_state(local_basis, σ))
    probs ./= sum(probs)
end
