export AbstractGibbsNetwork
export low_energy_spectrum, branch_state, bound_solution, merge_branches
export Solution
using Memoize

struct Solution
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
    probabilities::Vector{Float64}
    degeneracy::Vector{Int}
    largest_discarded_probability::Float64
end


empty_solution() = Solution([0.], [[]], [1.], -Inf)

function branch_state(network, σ)
    node = node_from_index(network, length(σ) + 1)
    basis = collect(1:length(local_energy(network, node)))
    vcat.(Ref(σ), basis)
end


function branch_solution(partial_sol::Solution, network::AbstractGibbsNetwork)

    Solution(
        vcat(
            [
                (en .+ update_energy(network, state))
                for (en, state) ∈ zip(partial_sol.energies, partial_sol.states)
            ]
            ...
        ),
        vcat(branch_state.(Ref(network), partial_sol.states)...),
        vcat(
            partial_sol.probabilities .* conditional_probability.(Ref(network), partial_sol.states)
            ...
        ),
        repeat(partial_sol.degeneracy, inner=local_dim),
        partial_sol.largest_discarded_probability
    )
end

function merge_branches(energy_atol::Float64)
    function _merge(network, partial_sol::Solution)
        node = node_from_index(network, length(partial_sol.states[1])+1)
        #boundaries = generate_boundary_states(network, partial_sol.states, node)
        boundaries = hcat(
            generate_boundary_states(network, partial_sol.states, node)...
        )

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

            best_idx = argmin(energies[start:stop]) # Use view if it works

            push!(new_energies, energies[start-1+best_idx])
            push!(new_states, states[start-1+best_idx])
            push!(new_probs, probs[start-1+best_idx])
            push!(new_degeneracy, degeneracy[start-1+best_idx])

            start = stop + 1
        end

        Solution(new_energies, new_states, new_probs, new_degeneracy, partial_sol.largest_discarded_probability)
    # 1. Znajdź boundary dla każdego solution
    # 2. Pogrupuj solution po boundary
    # 3. Iteruj po grupach:
    #    3a) w każdej grupie wybierz stan o najmniejszej energii
    #    3b) wybierz stany o tej samej energii i zsumuj ich degenerację -> to jest nowa degeneracja
    end
    _merge
end

function no_merge(partial_sol)
    partial_sol
end

function bound_solution(network, partial_sol::Solution, max_states::Int, merge_strategy=no_merge)
    partial_sol = merge_strategy(network, partial_sol)
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

    Solution(
        partial_sol.energies[indices],
        partial_sol.states[indices],
        partial_sol.probabilities[indices],
        partial_sol.degeneracy[indices],
        new_max_discarded_prob
    )
end


#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(network::AbstractGibbsNetwork, max_states::Int, merge_strategy=no_merge)
    sol = empty_solution()

    for _ ∈ 1:nv(network_graph(network))
        sol = bound_solution(network, branch_solution(sol, network), max_states, merge_strategy)
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
