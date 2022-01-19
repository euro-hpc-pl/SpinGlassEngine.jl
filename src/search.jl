export AbstractGibbsNetwork, low_energy_spectrum, branch_state, bound_solution
export merge_branches, Solution, SearchParameters
export exact_marginal_probability, exact_conditional_probabilities

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

function branch_energy(peps::PEPSNetwork{T, S}, eσ::Tuple{<:Real, Vector{Int}}) where {T, S}
    eσ[begin] .+ update_energy(peps, eσ[end])
end

function branch_state(network::PEPSNetwork{T, S}, σ::Vector{Int}) where {T, S}
    node = node_from_index(network, length(σ) + 1)
    vcat.(Ref(σ), collect(1:length(local_energy(network, node))))
end

# TODO: write functions: exact_marginal_probability, exact_conditional_probabilities
function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real, Vector{Int}}) where T
    #println("pσ[begin] ", exp(pσ[begin]))
    #println("conditional prob ", conditional_probability(ctr, pσ[end]))
    #exact_marginal_prob = exact_marginal_probability(ctr, pσ[end])
    #println("exact_marginal_prob ", exact_marginal_prob)
    #println("pσ[begin] ", exp(pσ[begin]))
    pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))
    #exact_marginal_prob = exact_marginal_probability(ctr, pσ[end])  # to compare with pσ[begin]
    # exact_cond_probs = exact_conditional_probabilities(ctr, pσ[end])
    # to compare with conditional_probability(ctr, pσ[end])
end

function exact_marginal_probability(fg, pσ::Vector{Int})# where T
    fg = factor_graph(ctr.peps)
    ig_states = decode_factor_graph_state.(Ref(fg), pσ)
    E =  energy(fg, ig_states)

    #states = Spectrum(ig).states
    #E = Spectrum(ig).energies
    P = exp.(-1 * E) #ctr.betas
    P = P./sum(P)
    st = [s[1:length(pσ)] for s in states]
    ind = findall(==(pσ), st)
    sum(P[ind])
    #println("prob ", sum(P[ind]))
end

function exact_conditional_probabilities(ig, pσ::Vector{Int})
    states = Spectrum(ig).states
    E = Spectrum(ig).energies
    P = exp.(-1 * E)
    P = P./sum(P)
    st = [s[1:length(pσ)] for s in states]
    ind = findall(==(pσ), st)
    st2 = [s[1:length(pσ)+1] for s in states]
    ind2 = findall(==(pσ), st2)
    println("cond ", [sum(P[ind]), P[ind2]])
    [sum(P[ind]), P[ind2]]
end

#=
marginal
p = exp.()=beta*E
p = p/sum(p)
ind = stan[1:len(sigma)] = sigma
return sum(p[ind])
potem obrócić
reverse_label_map w fg
for i= 1:N
rotated_states[:, ri] = state[:,i]

conditional
do ind a potem sumuję leng(sigma)+1
=#

function discard_probabilities(psol::Solution, cut_off_prob::Real)
    pcut = maximum(psol.probabilities) + log(cut_off_prob)
    Solution(psol, findall(p -> p >= pcut, psol.probabilities))
end

function branch_solution(psol::Solution, ctr::T, δprob::Real) where T <: AbstractContractor
    node = node_from_index(ctr.peps, length(psol.states[begin]) + 1)
    discard_probabilities(
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
    function _merge(psol::Solution)
        node = node_from_index(network, length(psol.states[1])+1)
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
    @showprogress "Search: " for _ ∈ 1:nv(factor_graph(ctr.peps))
        sol = branch_solution(sol, ctr, sparams.cut_off_prob)
        sol = bound_solution(sol, sparams.max_states, merge_strategy)
        # _clear_cache(network, sol) # TODO: make it work properly
    end

    # Translate variable order (from network to factor graph)
    inner_perm = sortperm([
        factor_graph(ctr.peps).reverse_label_map[idx]
        for idx ∈ ctr.peps.vertex_map.(iteration_order(ctr.peps))
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
