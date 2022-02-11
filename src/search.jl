export
       low_energy_spectrum,
       branch_state,
       bound_solution,
       merge_branches,
       Solution,
       SearchParameters,
       exact_marginal_probability,
       exact_conditional_probabilities

struct SearchParameters
    max_states::Int
    cut_off_prob::Real

    function SearchParameters(max_states::Int=1, cut_off_prob::Real=0.0)
        new(max_states, cut_off_prob)
    end
end
struct Solution
    energies::Vector{<:Real}
    states::Vector{Vector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
end
empty_solution() = Solution([0.0], [Vector{Int}[]], [0.0], [1], -Inf)

function Solution(
    sol::Solution, idx::Vector{Int}, ldp::Real=sol.largest_discarded_probability
)
    Solution(
        sol.energies[idx],
        sol.states[idx],
        sol.probabilities[idx],
        sol.degeneracy[idx],
        ldp
    )
end

function branch_energy(ctr::MpsContractor{T}, eσ::Tuple{<:Real, Vector{Int}}) where {T, S}
    eσ[begin] .+ update_energy(ctr.peps, eσ[end])
end

function branch_state(ctr::MpsContractor{T}, σ::Vector{Int}) where {T, S}
    node = ctr.iteration_order[length(σ) + 1]
    vcat.(Ref(σ), collect(1:length(local_energy(ctr.peps, node))))
end

function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real, Vector{Int}}) where T
    #exact_marginal_prob = exact_marginal_probability(ctr, pσ[end])
    #@assert exact_marginal_prob ≈ exp(pσ[begin])
    #exact_cond_probs = exact_conditional_probabilities(ctr, pσ[end])
    #@assert exact_cond_probs ≈ conditional_probability(ctr, pσ[end])
    #cond_prob = conditional_probability(ctr, pσ[end])
    prob = pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))

    #if !(exact_cond_probs ≈ conditional_probability(ctr, pσ[end]))
    #    println("ERROR")
    #end
    #println(pσ)
    #@infiltrate
    prob
    #pσ[begin] .+ log.(exact_cond_probs)
end

@memoize function spectrum(factor_graph::LabelledGraph{S, T}) where {S, T}
    ver = vertices(factor_graph)
    rank = cluster_size.(Ref(factor_graph), ver)
    states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
    energy.(Ref(factor_graph), states), states
end

function exact_marginal_probability(
    ctr::MpsContractor{T},
    σ::Vector{Int}
) where T
    target_state = decode_state(ctr.peps, σ, true)
    energies, states = spectrum(ctr.peps.factor_graph)

    prob = exp.(-ctr.betas[end] .* energies)
    prob ./= sum(prob)
    sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
end

function exact_conditional_probabilities(ctr::MpsContractor{T}, σ::Vector{Int}) where T
    probs = exact_marginal_probability.(Ref(ctr), branch_state(ctr.peps, σ))
    probs ./= sum(probs)
end

function discard_probabilities(psol::Solution, cut_off_prob::Real)
    pcut = maximum(psol.probabilities) + log(cut_off_prob)
    Solution(psol, findall(p -> p >= pcut, psol.probabilities))
end

function branch_solution(psol::Solution, ctr::T, δprob::Real) where T <: AbstractContractor
    node = ctr.iteration_order[length(psol.states[begin]) + 1]
    num_states = cluster_size(ctr.peps, node)
    discard_probabilities(
        Solution(
            vcat(branch_energy.(Ref(ctr), zip(psol.energies, psol.states))...),
            vcat(branch_state.(Ref(ctr), psol.states)...),
            vcat(branch_probability.(Ref(ctr), zip(psol.probabilities, psol.states))...),
            repeat(psol.degeneracy, inner=num_states),
            psol.largest_discarded_probability
        ),
        δprob
    )
end

function merge_branches(ctr::MpsContractor{T}) where {T}
    function _merge(psol::Solution)
        network = ctr.peps
        node = ctr.iteration_order[length(psol.states[1])+1]
        boundaries = hcat(boundary_state.(Ref(network), psol.states, Ref(node))...)'
        _, indices = SpinGlassNetworks.unique_dims(boundaries, 1)

        sorting_idx = sortperm(indices)
        sorted_indices = indices[sorting_idx]
        nsol = Solution(psol, Vector{Int}(sorted_indices)) #TODO Vector{Int} should be rm

        energies = typeof(nsol.energies[begin])[]
        states = typeof(nsol.states[begin])[]
        probs = typeof(nsol.probabilities[begin])[]
        degeneracy = typeof(nsol.degeneracy[begin])[]

        start = 1
        while start <= size(boundaries, 1)
            stop = start
            bsize = size(boundaries, 1)
            while stop + 1 <= bsize && sorted_indices[start] == sorted_indices[stop+1]
                stop = stop + 1
            end
            best_idx = argmin(@view nsol.energies[start:stop]) + start - 1

            push!(energies, nsol.energies[best_idx])
            push!(states, nsol.states[best_idx])
            push!(probs, nsol.probabilities[best_idx])
            push!(degeneracy, nsol.degeneracy[best_idx])
            start = stop + 1
        end
        Solution(energies, states, probs, degeneracy, psol.largest_discarded_probability)
    end
    _merge
end
no_merge(partial_sol::Solution) = partial_sol

function bound_solution(psol::Solution, max_states::Int, merge_strategy=no_merge)
    if length(psol.probabilities) <= max_states
        probs = vcat(psol.probabilities, -Inf)
        k = length(probs)
    else
        probs = psol.probabilities
        k = max_states + 1
    end
    idx = partialsortperm(probs, 1:k, rev=true)
    ldp = max(psol.largest_discarded_probability, probs[idx[end]])
    merge_strategy(Solution(psol, idx[1:k-1], ldp))
end

#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(
    ctr::T, sparams::SearchParameters, merge_strategy=no_merge
) where T <: AbstractContractor
    # Build all boundary mps
    @showprogress "Preprocesing: " for i ∈ ctr.peps.nrows:-1:1 dressed_mps(ctr, i) end

    # Start branch and bound search
    sol = empty_solution()
    @showprogress "Search: " for node ∈ ctr.iteration_order
        sol = branch_solution(sol, ctr, sparams.cut_off_prob)
        sol = bound_solution(sol, sparams.max_states, merge_strategy)
        # _clear_cache(network, sol) # TODO: make it work properly
    end

    # Translate variable order (from network to factor graph)
    inner_perm = sortperm([
        ctr.peps.factor_graph.reverse_label_map[idx]
        for idx ∈ ctr.peps.vertex_map.(ctr.iteration_order)
    ])

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    sol = Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability
    )

    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.factor_graph), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol
end
