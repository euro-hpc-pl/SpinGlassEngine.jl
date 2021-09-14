export 
    AbstractGibbsNetwork,
    low_energy_spectrum, 
    branch_state,
    bound_solution,
    merge_branches,
    Solution,
    Memoize

struct Solution
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
    probabilities::Vector{Float64}
    degeneracy::Vector{Int}
    largest_discarded_probability::Float64
end


function _clear_cache(network::AbstractGibbsNetwork, sol::Solution)
    i, j = node_from_index(network, length(first(sol.states))+1)
    if j != network.ncols return end
    delete!(memoize_cache(mps), (network, i))
    delete!(memoize_cache(dressed_mps), (network, i))
    delete!(memoize_cache(mpo), (network, i-1))
    delete!(memoize_cache(_mpo), (network, i-1))
    lec = memoize_cache(left_env)
    delete!.(Ref(lec), filter(k->k[2]==i, keys(lec)))
    rec = memoize_cache(right_env)
    delete!.(Ref(rec), filter(k->k[2]==i, keys(rec)))
end


empty_solution() = Solution([0.], [[]], [1.], [1], -Inf)


function branch_state(network, σ)
    node = node_from_index(network, length(σ) + 1)
    basis = collect(1:length(local_energy(network, node)))
    vcat.(Ref(σ), basis)
end


function branch_solution(partial_sol::Solution, network::AbstractGibbsNetwork)
    local_dim = length(local_energy(network, node_from_index(network, length(partial_sol.states[1])+1))) 
    Solution(
        vcat
        (
            [
                (en .+ update_energy(network, state))
                for (en, state) ∈ zip(partial_sol.energies, partial_sol.states)
            ]
            ...
        ),
        vcat(branch_state.(Ref(network), partial_sol.states)...),
        vcat(
            [
                partial_sol.probabilities[i] .+ log.(p) 
                for (i,p) in enumerate(conditional_probability.(Ref(network), partial_sol.states))
                    ]
            ...
        ),
        repeat(partial_sol.degeneracy, inner=local_dim),
        partial_sol.largest_discarded_probability
    )
end

function merge_branches(
    network::AbstractGibbsNetwork{S, T}, 
    energy_atol::Float64
) where {S, T}
    function _merge(partial_sol::Solution)
        node = node_from_index(network, length(partial_sol.states[1])+1)
        boundaries = hcat(boundary_state.(Ref(network), partial_sol.states, Ref(node))...)'  
        _unique_boundaries, indices = SpinGlassNetworks.unique_dims(boundaries, 1)

        sorting_idx = sortperm(indices)
        sorted_indices = indices[sorting_idx]
        start = 1

        energies = partial_sol.energies[sorting_idx]
        states = partial_sol.states[sorting_idx]
        probs = partial_sol.probabilities[sorting_idx]
        degeneracy = partial_sol.degeneracy[sorting_idx]

        new_energies = []
        new_states = []
        new_probs = []
        new_degeneracy = []

        while start <= size(boundaries, 1)
            stop = start
            while stop + 1 <= size(boundaries, 1) && sorted_indices[start] == sorted_indices[stop+1]
                stop = stop + 1
            end

            best_idx = argmin(@view energies[start:stop]) + start-1

            push!(new_energies, energies[best_idx])
            push!(new_states, states[best_idx])
            push!(new_probs, probs[best_idx])
            push!(new_degeneracy, degeneracy[best_idx])

            start = stop + 1
        end
        Solution(new_energies, new_states, new_probs, new_degeneracy, partial_sol.largest_discarded_probability)
    end
    _merge
end

function no_merge(partial_sol::Solution)
    partial_sol
end

function bound_solution(partial_sol::Solution, max_states::Int, merge_strategy=no_merge)
    if length(partial_sol.probabilities) <= max_states
        probs = vcat(partial_sol.probabilities, -Inf)
        k = length(probs)
    else
        probs = partial_sol.probabilities
        k = max_states + 1
    end

    indices = partialsortperm(probs, 1:k, rev=true)
    new_max_discarded_prob = max(partial_sol.largest_discarded_probability, probs[indices[end]])

    indices = @view indices[1:k-1]

    merge_strategy(Solution(
        partial_sol.energies[indices],
        partial_sol.states[indices],
        partial_sol.probabilities[indices],
        partial_sol.degeneracy[indices],
        new_max_discarded_prob
    ))
end


#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(network::AbstractGibbsNetwork, max_states::Int, merge_strategy=no_merge)
    # Build all boundary mps
    @showprogress "Preprocesing: " for i ∈ network.nrows:-1:1 dressed_mps(network, i) end

    # Start branch and bound search
    sol = empty_solution()
    @showprogress "Search: " for _ ∈ 1:nv(network_graph(network))
        sol = branch_solution(sol, network)
        sol = bound_solution(sol, max_states, merge_strategy)
        _clear_cache(network, sol) 
    end

    # Translate variable order (from network to factor graph)
    inner_perm = sortperm([
        factor_graph(network).reverse_label_map[idx]
        for idx ∈ network.vertex_map.(iteration_order(network))
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
