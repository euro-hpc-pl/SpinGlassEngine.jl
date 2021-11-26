export AbstractGibbsNetwork, low_energy_spectrum, branch_state, bound_solution
export merge_branches, Solution, SearchParameters

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
empty_solution() = Solution([0.0], [Vector{Int}[]], [1.0], [1], -Inf)

function branch_energy(peps::PEPSNetwork{T, S}, eσ::Tuple{<:Real, Vector{Int}}) where {T, S}
    eσ[begin] .+ update_energy(peps, eσ[end])
end

function branch_state(network::PEPSNetwork{T, S}, σ::Vector{Int}) where {T, S}
    node = node_from_index(network, length(σ) + 1)
    vcat.(Ref(σ), collect(1:length(local_energy(network, node))))
end

function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real, Vector{Int}}) where T
    pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))
end

function _discard_probabilities(partial_sol::Solution, cut_off_prob::Real)
    prob = partial_sol.probabilities
    idx = findall(p -> p >= maximum(prob) + log(cut_off_prob), prob)
    Solution(
        partial_sol.energies[idx],
        partial_sol.states[idx],
        prob[idx],
        partial_sol.degeneracy[idx],
        partial_sol.largest_discarded_probability
    )
end

function branch_solution(psol::Solution, ctr::T, δprob::Real) where T <: AbstractContractor
    node = node_from_index(ctr.peps, length(psol.states[begin]) + 1)
    
    _discard_probabilities(
        Solution(
            vcat(branch_energy.(Ref(ctr.peps), zip(psol.energies, psol.states))...),
            vcat(branch_state.(Ref(ctr.peps), psol.states)...),
            vcat(branch_probability.(Ref(ctr), zip(psol.probabilities, psol.states))...),
            repeat(psol.degeneracy, inner=length(local_energy(ctr.peps, node))),
            psol.largest_discarded_probability
        ),
        δprob
    )
end

function merge_branches(network::AbstractGibbsNetwork{S, T}) where {S, T}
    function _merge(partial_sol::Solution)
        node = node_from_index(network, length(partial_sol.states[1])+1)
        boundaries = hcat(boundary_state.(Ref(network), partial_sol.states, Ref(node))...)'
        _, indices = SpinGlassNetworks.unique_dims(boundaries, 1)

        sorting_idx = sortperm(indices)
        sorted_indices = indices[sorting_idx]
        start = 1

        energies = partial_sol.energies[sorting_idx]
        states = partial_sol.states[sorting_idx]
        probs = partial_sol.probabilities[sorting_idx]
        degeneracy = partial_sol.degeneracy[sorting_idx]

        new_energies = typeof(energies[begin])[]
        new_states = typeof(states[begin])[]
        new_probs = typeof(probs[begin])[]
        new_degeneracy = typeof(degeneracy[begin])[]

        while start <= size(boundaries, 1)
            stop = start
            bsize = size(boundaries, 1)
            while stop + 1 <= bsize && sorted_indices[start] == sorted_indices[stop+1]
                stop = stop + 1
            end

            best_idx = argmin(@view energies[start:stop]) + start-1

            push!(new_energies, energies[best_idx])
            push!(new_states, states[best_idx])
            push!(new_probs, probs[best_idx])
            push!(new_degeneracy, degeneracy[best_idx])

            start = stop + 1
        end

        Solution(
            new_energies,
            new_states,
            new_probs,
            new_degeneracy,
            partial_sol.largest_discarded_probability
        )
    end
    _merge
end
no_merge(partial_sol::Solution) = partial_sol

function bound_solution(partial_sol::Solution, max_states::Int, merge_strategy=no_merge)
    if length(partial_sol.probabilities) <= max_states
        probs = vcat(partial_sol.probabilities, -Inf)
        k = length(probs)
    else
        probs = partial_sol.probabilities
        k = max_states + 1
    end

    indices = partialsortperm(probs, 1:k, rev=true)
    new_max_discarded_prob = max(
        partial_sol.largest_discarded_probability, probs[indices[end]]
    )
    indices = @view indices[1:k-1]

    merge_strategy(
        Solution(
            partial_sol.energies[indices],
            partial_sol.states[indices],
            partial_sol.probabilities[indices],
            partial_sol.degeneracy[indices],
            new_max_discarded_prob
        )
    )
end

#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(
    contractor::T, sparams::SearchParameters, merge_strategy=no_merge
) where T <: AbstractContractor
    # Build all boundary mps
    @showprogress "Preprocesing: " for i ∈ contractor.peps.nrows:-1:1
        dressed_mps(contractor, i)
    end

    # Start branch and bound search
    sol = empty_solution()
    @showprogress "Search: " for _ ∈ 1:nv(factor_graph(contractor.peps))
        sol = branch_solution(sol, contractor, sparams.cut_off_prob)
        sol = bound_solution(sol, sparams.max_states, merge_strategy)
        # _clear_cache(network, sol) # TODO: make it work properly
    end

    # Translate variable order (from network to factor graph)
    inner_perm = sortperm([
        factor_graph(contractor.peps).reverse_label_map[idx]
        for idx ∈ contractor.peps.vertex_map.(iteration_order(contractor.peps))
    ])

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability
    )
end
