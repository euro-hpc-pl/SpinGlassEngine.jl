export
       SearchParameters,
       merge_branches,
       merge_branches_blur,
       low_energy_spectrum,
       Solution,
       bound_solution,
       gibbs_sampling,
       NoDroplets,
       Droplet,
       SingleLayerDroplets,
       Flip,
       unpack_droplets,
       decode_to_spin,
       hamming_distance,
       empty_solution,
       branch_energy,
       no_merge

"""
$(TYPEDSIGNATURES)
A struct representing search parameters for low-energy spectrum search.

## Fields 
- `max_states::Int`: The maximum number of states to be considered during the search. Default is 1, indicating a single state search.
- `cut_off_prob::Real`: The cutoff probability for terminating the search. Default is 0.0, meaning no cutoff based on probability.
    
SearchParameters encapsulates parameters that control the behavior of low-energy spectrum search algorithms in the SpinGlassPEPS package. 
Users can customize these parameters to adjust the search strategy and resource usage according to their specific needs.
    
"""
struct SearchParameters
    max_states::Int
    cut_off_prob::Real

    function SearchParameters(max_states::Int=1, cut_off_prob::Real=0.0)
        new(max_states, cut_off_prob)
    end
end

struct NoDroplets end

Base.iterate(drop::NoDroplets) = nothing
Base.copy(s::NoDroplets) = s

"""
A data structure representing the properties and criteria for single-layer droplets in the context of the SpinGlassPEPS package.

A `SingleLayerDroplets` object is used to specify the maximum energy, minimum size, and metric for single-layer droplets in the SpinGlassPEPS system.
    
## Fields
- `max_energy::Real`: The maximum allowed excitation energy for single-layer droplets. It is typically a real number.
- `min_size::Int`: The minimum size (number of sites) required for a single-layer droplet to be considered significant.
- `metric::Symbol`: The metric used to evaluate the significance of a single-layer droplet. 
This can be `:no_metric` or other custom metrics defined in the package.
    
## Constructors
- `SingleLayerDroplets(max_energy::Real = 1.0, min_size::Int = 1, metric::Symbol = :no_metric)`: Creates a new 
`SingleLayerDroplets` object with the specified maximum energy, minimum size, and metric.
    
"""
struct SingleLayerDroplets
    max_energy :: Real
    min_size :: Int
    metric :: Symbol 
    SingleLayerDroplets(max_energy = 1.0, min_size = 1, metric = :no_metric) = new(max_energy, min_size, metric)
end

"""
A data structure representing a set of flips or changes in states for nodes in the SpinGlassPEPS package.

A `Flip` object contains information about the support, state changes, and spinxor values for a set of node flips in the SpinGlassPEPS system.
    
## Fields
- `support::Vector{Int}`: An array of integers representing the indices of nodes where flips occur.
- `state::Vector{Int}`: An array of integers representing the new states for the nodes in the `support`.
- `spinxor::Vector{Int}`: An array of integers representing the spin-xor values for the nodes in the `support`.
    
## Constructors  
- `Flip(support::Vector{Int}, state::Vector{Int}, spinxor::Vector{Int})`: 
Creates a new `Flip` object with the specified support, state changes, and spinxor values.
    
"""
struct Flip
    support :: Vector{Int}
    state :: Vector{Int}
    spinxor :: Vector{Int}
end

"""
$(TYPEDSIGNATURES)
A data structure representing a droplet in the context of the SpinGlassPEPS package.
A `Droplet` represents an excitation in the SpinGlassPEPS system. It contains information about the excitation energy, 
the site where the droplet starts, the site where it ends, the states of nodes flipped by the droplet, 
and any sub-droplets on top of the current droplet.

## Fields
- `denergy::Real`: The excitation energy of the droplet, typically a real number.
- `first::Int`: The site index where the droplet starts.
- `last::Int`: The site index where the droplet ends.
- `flip::Flip`: The states of nodes flipped by the droplet, often represented using a `Flip` type.
- `droplets::Union{NoDroplets, Vector{Droplet}}`: A field that can be either `NoDroplets()` if there are no sub-droplets 
on top of the current droplet or a vector of `Droplet` objects representing sub-droplets. 
This field may be used to build a hierarchy of droplets in more complex excitations.

"""
mutable struct Droplet
    denergy :: Real  # excitation energy
    first :: Int  # site where droplet starts
    last :: Int  
    flip :: Flip  # states of nodes flipped by droplet
    droplets :: Union{NoDroplets, Vector{Droplet}}  # subdroplets on top of the current droplet; can be empty
end

Droplets = Union{NoDroplets, Vector{Droplet}}  # Can't be defined before Droplet struct

Base.getindex(s::NoDroplets, ::Any) = NoDroplets()

"""
$(TYPEDSIGNATURES)
A struct representing a solution obtained from a low-energy spectrum search.

## Fields
- `energies::Vector{<:Real}`: A vector containing the energies of the discovered states.
- `states::Vector{Vector{Int}}`: A vector of state configurations corresponding to the energies.
- `probabilities::Vector{<:Real}`: The probabilities associated with each discovered state.
- `degeneracy::Vector{Int}`: The degeneracy of each energy level.
- `largest_discarded_probability::Real`: The probability of the largest discarded state.
- `droplets::Vector{Droplets}`: A vector of droplets associated with each state.
- `spins::Vector{Vector{Int}}`: The spin configurations corresponding to each state.
    
The `Solution` struct holds information about the results of a low-energy spectrum search, including the energy levels, 
state configurations, probabilities, degeneracy, and additional details such as droplets and spin configurations. 
Users can access this information to analyze and interpret the search results. 
"""
struct Solution
    energies::Vector{<:Real}
    states::Vector{Vector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
    droplets::Vector{Droplets}
    spins::Vector{Vector{Int}}
end

"""
$(TYPEDSIGNATURES)
Create an empty `Solution` object with a specified number of states.

This function creates an empty `Solution` object with the given number of states, initializing its fields with default values.

## Arguments
- `n::Int`: The number of states for which the `Solution` object is created.

## Returns
An empty `Solution` object with default field values, ready to store search results for a specified number of states.
"""
@inline empty_solution(n::Int=1) = Solution(zeros(n), fill(Vector{Int}[], n), zeros(n), ones(Int, n),
                                            -Inf, repeat([NoDroplets()], n), fill(Vector{Int}[], n))

"""
$(TYPEDSIGNATURES)
Create a new `Solution` object by selecting specific states from an existing `Solution`.

This constructor allows you to create a new `Solution` object by selecting specific states from an existing `Solution`. 
It copies the energies, states, probabilities, degeneracy, droplets, and spins information for the selected states 
while allowing you to set a custom `largest_discarded_probability`.

## Arguments
- `sol::Solution`: The original `Solution` object from which states are selected.
- `idx::Vector{Int}`: A vector of indices specifying the states to be selected from the original `Solution`.
- `ldp::Real=sol.largest_discarded_probability`: (Optional) The largest discarded probability for the new `Solution`. 
By default, it is set to the largest discarded probability of the original `Solution`.

## Returns
A new `Solution` object containing information only for the selected states.
"""
function Solution(
    sol::Solution, idx::Vector{Int}, ldp::Real=sol.largest_discarded_probability
)
    Solution(
        sol.energies[idx],
        sol.states[idx],
        sol.probabilities[idx],
        sol.degeneracy[idx],
        ldp,
        sol.droplets[idx],
        sol.spins[idx]
    )
end

"""
$(TYPEDSIGNATURES)
Calculates the energy contribution of a branch given a base energy and a spin configuration.

This function calculates the energy contribution of a branch in a SpinGlassPEPS calculation. 
It takes a `MpsContractor` object `ctr` and a tuple `eσ` containing a base energy as the first element 
and a spin configuration represented as a vector of integers as the second element. 
The function calculates the branch energy by adding the base energy to the energy contribution 
of the given spin configuration obtained from the `update_energy` function.

## Arguments
- `ctr::MpsContractor{T}`: An instance of the `MpsContractor` type parameterized by the strategy type `T`.
- `eσ::Tuple{<:Real, Vector{Int}}`: A tuple containing the base energy as the first element (a real number) 
and the spin configuration as the second element (a vector of integers).

## Returns
The branch energy, which is the sum of the base energy and the energy contribution of the spin configuration.

"""
@inline function branch_energy(ctr::MpsContractor{T}, eσ::Tuple{<:Real, Vector{Int}}) where T
    eσ[begin] .+ update_energy(ctr, eσ[end])
end

# """
# $(TYPEDSIGNATURES)
# """
@inline function branch_energies(ctr::MpsContractor{T}, psol::Solution) where T
    reduce(vcat, branch_energy.(Ref(ctr), zip(psol.energies, psol.states)))
end

# """
# $(TYPEDSIGNATURES)
# """
function branch_states(local_basis::Vector{Int}, vec_states::Vector{Vector{Int}})
    states = reduce(hcat, vec_states)
    num_states = length(local_basis)
    lstate, nstates = size(states)
    ns = Array{Int}(undef, lstate+1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate+1, :, :] .= reshape(local_basis, num_states, 1, 1)
    collect(eachcol(reshape(ns, lstate+1, nstates * num_states)))
end

# """
# $(TYPEDSIGNATURES)
# """
function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real, Vector{Int}}) where T
    pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))
end

# """
# $(TYPEDSIGNATURES)
# """
function discard_probabilities!(psol::Solution, cut_off_prob::Real)
    pcut = maximum(psol.probabilities) + log(cut_off_prob)
    if minimum(psol.probabilities) >= pcut return psol end
    local_ldp = maximum(psol.probabilities[psol.probabilities .< pcut])
    ldp = max(local_ldp, psol.largest_discarded_probability)
    Solution(psol, findall(p -> p >= pcut, psol.probabilities), ldp)
end

# """
# $(TYPEDSIGNATURES)
# """
function local_spins(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    spectrum(network, vertex).states_int
end

# """
# $(TYPEDSIGNATURES)
# """
function branch_solution(psol::Solution, ctr::T) where T <: AbstractContractor
    num_states = cluster_size(ctr.peps, ctr.current_node)
    basis_states = collect(1:num_states)
    basis_spins = local_spins(ctr.peps, ctr.current_node)
    boundaries = boundary_states(ctr, psol.states, ctr.current_node)
    Solution(
        branch_energies(ctr, psol),
        branch_states(basis_states, psol.states),
        reduce(vcat, branch_probability.(Ref(ctr), zip(psol.probabilities, boundaries))),
        repeat(psol.degeneracy, inner=num_states),
        psol.largest_discarded_probability,
        repeat(psol.droplets, inner=num_states),#,
        branch_states(basis_spins, psol.spins)
    )
end

# """
# $(TYPEDSIGNATURES)
# """
(method::NoDroplets)(
    ctr::MpsContractor{T}, 
    best_idx::Int, 
    energies::Vector{<:Real},  
    states::Vector{Vector{Int}}, 
    droplets::Vector{Droplets}, 
    spins::Vector{Vector{Int}}
    ) where T= NoDroplets()

function (method::SingleLayerDroplets)(
    ctr::MpsContractor{T}, 
    best_idx::Int, 
    energies::Vector{<:Real}, 
    states::Vector{Vector{Int}}, 
    droplets::Vector{Droplets}, 
    spins::Vector{Vector{Int}}
    ) where T
    ndroplets = copy(droplets[best_idx])
    bstate  = states[best_idx]
    benergy = energies[best_idx]
    bspin = spins[best_idx]

    for ind ∈ (1 : best_idx - 1..., best_idx + 1 : length(energies)...)
        flip_support = findall(bstate .!= states[ind])
        flip_state = states[ind][flip_support]
        flip_spinxor = bspin[flip_support] .⊻ spins[ind][flip_support]
        flip = Flip(flip_support, flip_state, flip_spinxor)
        denergy = energies[ind] - benergy
        droplet = Droplet(denergy, flip_support[1], length(bstate), flip, NoDroplets())
        if droplet.denergy <= method.max_energy && hamming_distance(droplet.flip) >= method.min_size
            ndroplets = my_push!(ndroplets, droplet, method)
        end
        for subdroplet ∈ droplets[ind]
            new_droplet = merge_droplets(method, droplet, subdroplet)
            if new_droplet.denergy <= method.max_energy && hamming_distance(new_droplet.flip) >= method.min_size
                ndroplets = my_push!(ndroplets, new_droplet, method)
            end
        end
    end
    if typeof(ndroplets) == NoDroplets
        return ndroplets
    else
        return filter_droplets(ndroplets, method)
    end
end

function filter_droplets(all_droplets::Vector{Droplet}, method::SingleLayerDroplets)
    sorted_droplets = sort(all_droplets, by = droplet -> (droplet.denergy))
    if method.metric == :hamming
        cutoff = method.min_size
    else #method.metric == :no_metric
        cutoff = -Inf
    end
    
    filtered_droplets = Droplet[]
    for droplet in sorted_droplets
        should_push = true
        for existing_drop in filtered_droplets
            if diversity_metric(existing_drop, droplet, method.metric) < cutoff
                should_push = false
                break
            end
        end
        if should_push
            push!(filtered_droplets, droplet)
        end
    end
    
    filtered_droplets
end

function my_push!(ndroplets::Droplets, droplet::Droplet, method)
    if typeof(ndroplets) == NoDroplets
        ndroplets = Droplet[]
    end
    push!(ndroplets, droplet)
    ndroplets
end

function diversity_metric(drop1::Droplet, drop2::Droplet, metric::Symbol)
    if metric == :hamming
        d = hamming_distance(drop1.flip, drop2.flip)
    else 
        d = Inf
    end
    d
end

function hamming_distance(flip::Flip)
    sum(count_ones(st) for st ∈ flip.spinxor)
end

function hamming_distance(flip1::Flip, flip2::Flip)
    n1, n2, hd = 1, 1, 0
    l1, l2 = length(flip1.support), length(flip2.support)
    while (n1 <= l1) && (n2 <= l2)
        if flip1.support[n1] == flip2.support[n2]
            if flip1.state[n1] != flip2.state[n2]
                hd += count_ones(flip1.spinxor[n1] ⊻ flip2.spinxor[n2])
            end
            n1 += 1
            n2 += 1
        elseif flip1.support[n1] < flip2.support[n2]
            hd += count_ones(flip1.spinxor[n1])
            n1 += 1
        else
            hd += count_ones(flip2.spinxor[n2])
            n2 += 1
        end
    end
    while n1 <= l1
        hd += count_ones(flip1.spinxor[n1])
        n1 += 1
    end
    while n2 <= l2
        hd += count_ones(flip2.spinxor[n2])
        n2 += 1
        end
    hd
end

function merge_droplets(method::SingleLayerDroplets, droplet::Droplet, subdroplet::Droplet)
    denergy = droplet.denergy + subdroplet.denergy
    first = min(droplet.first, subdroplet.first)
    last = max(droplet.last, subdroplet.last)

    flip = droplet.flip
    subflip = subdroplet.flip

    i1, i2, i3 = 1, 1, 1
    ln = length(union(flip.support, subflip.support))

    new_support = zeros(Int, ln)
    new_state = zeros(Int, ln)
    new_spinxor = zeros(Int, ln)

    while i1 <= length(flip.support) &&  i2 <= length(subflip.support)
        if flip.support[i1] == subflip.support[i2]
            new_support[i3] = flip.support[i1]
            new_state[i3] = subflip.state[i2]
            new_spinxor[i3] = flip.spinxor[i1] ⊻ subflip.spinxor[i2]
            i1 += 1
            i2 += 1
            i3 += 1
        elseif flip.support[i1] < subflip.support[i2]
            new_support[i3] = flip.support[i1]
            new_state[i3] = flip.state[i1]
            new_spinxor[i3] = flip.spinxor[i1]
            i1 += 1
            i3 += 1
        else # flip.support[i1] > subflip.support[i2]
            new_support[i3] = subflip.support[i2]
            new_state[i3] = subflip.state[i2]
            new_spinxor[i3] = subflip.spinxor[i2]
            i2 += 1
            i3 += 1
        end
    end
    while i1 <= length(flip.support)
        new_support[i3] = flip.support[i1]
        new_state[i3] = flip.state[i1]
        new_spinxor[i3] = flip.spinxor[i1]
        i1 += 1
        i3 += 1
    end
    while i2 <= length(subflip.support)
        new_support[i3] = subflip.support[i2]
        new_state[i3] = subflip.state[i2]
        new_spinxor[i3] = subflip.spinxor[i2]
        i2 += 1
        i3 += 1
    end
    flip = Flip(new_support, new_state, new_spinxor)
    Droplet(denergy, first, last, flip, NoDroplets())
end


function flip_state(state::Vector{Int}, flip::Flip)
    new_state = copy(state)
    new_state[flip.support] .= flip.state
    new_state
end

function unpack_droplets(sol, β)  # have β in sol ?
    energies = typeof(sol.energies[begin])[]
    states = typeof(sol.states[begin])[]
    probs = typeof(sol.probabilities[begin])[]
    degeneracy = typeof(sol.degeneracy[begin])[]
    droplets = Droplets[]
    spins = typeof(sol.spins[begin])[]

    for i in 1:length(sol.energies)
        push!(energies, sol.energies[i])
        push!(states, sol.states[i])
        push!(probs, sol.probabilities[i])
        push!(degeneracy, 1)
        push!(droplets, NoDroplets())
        push!(spins, sol.spins[i])

        for droplet in sol.droplets[i]
            push!(energies, sol.energies[i] + droplet.denergy)
            push!(states, flip_state(sol.states[i], droplet.flip))
            push!(probs, sol.probabilities[i] - β * droplet.denergy)
            push!(degeneracy, 1)
            push!(droplets, NoDroplets())
            push!(spins, sol.spins[i])
        end
    end

    inds = sortperm(energies)
    Solution(
        energies[inds], 
        states[inds], 
        probs[inds], 
        degeneracy[inds], 
        sol.largest_discarded_probability, 
        droplets[inds], 
        spins[inds]
        )
end


"""
$(TYPEDSIGNATURES)
Merge branches of a contractor based on specified merge type and droplet update strategy.

This function merges branches of a contractor (`ctr`) based on a specified merge type (`merge_type`) 
and an optional droplet update strategy (`update_droplets`). 
It returns a function `_merge` that can be used to merge branches in a solution.

## Arguments
- `ctr::MpsContractor{T}`: A contractor for which branches will be merged.
- `merge_type::Symbol=:nofit`: (Optional) The merge type to use. Defaults to `:nofit`. Possible values are `:nofit`, `:fit`, and `:python`.
- `update_droplets=NoDroplets()`: (Optional) The droplet update strategy. Defaults to `NoDroplets()`. You can provide a custom droplet update strategy if needed.

## Returns
A function `_merge` that can be used to merge branches in a solution.

## Details
The `_merge` function can be applied to a `Solution` object to merge its branches based on the specified merge type and droplet update strategy.
"""
function merge_branches(ctr::MpsContractor{T}, merge_type::Symbol=:nofit, update_droplets=NoDroplets()) where {T}
    function _merge(psol::Solution)
        node = get(ctr.nodes_search_order, length(psol.states[1])+1, ctr.node_outside)
        boundaries = boundary_states(ctr, psol.states, node)
        _, bnd_types = SpinGlassNetworks.unique_dims(boundaries, 1)
        sorting_idx = sortperm(bnd_types)
        sorted_bnd_types = bnd_types[sorting_idx]
        nsol = Solution(psol, Vector{Int}(sorting_idx)) #TODO Vector{Int} should be rm
        energies = typeof(nsol.energies[begin])[]
        states = typeof(nsol.states[begin])[]
        spins = typeof(nsol.spins[begin])[]
        probs = typeof(nsol.probabilities[begin])[]
        degeneracy = typeof(nsol.degeneracy[begin])[]
        droplets = Droplets[]

        start = 1
        bsize = size(boundaries, 1)
        while start <= bsize
            stop = start
            while stop + 1 <= bsize && sorted_bnd_types[start] == sorted_bnd_types[stop+1]
                stop = stop + 1
            end
            best_idx_bnd = argmin(@view nsol.energies[start:stop])
            best_idx = best_idx_bnd + start - 1

            new_degeneracy = 0
            ind_deg = []
            for i in start:stop
                if nsol.energies[i] <= nsol.energies[best_idx] + 1E-12 # this is hack for now
                    new_degeneracy += nsol.degeneracy[i]
                    push!(ind_deg, i)
                end
            end

            if merge_type == :fit
                c = Statistics.median(
                    ctr.betas[end] .* nsol.energies[start:stop] .+ nsol.probabilities[start:stop]
                )
                new_prob = -ctr.betas[end] .* nsol.energies[best_idx] .+ c
                push!(probs, new_prob)
            elseif merge_type == :nofit
                push!(probs, nsol.probabilities[best_idx])
            elseif merge_type == :python
                push!(probs, Statistics.mean(nsol.probabilities[ind_deg]))
            end

            ## states with unique boundary => we take the one with best energy
            ## treat other states with the same boundary as droplets on top of the best one
            excitation = update_droplets(ctr, best_idx_bnd, nsol.energies[start:stop], nsol.states[start:stop],
                                        nsol.droplets[start:stop], nsol.spins[start:stop])
            push!(droplets, excitation)

            push!(energies, nsol.energies[best_idx])
            push!(states, nsol.states[best_idx])
            push!(degeneracy, new_degeneracy)
            push!(spins, nsol.spins[best_idx])
            start = stop + 1
        end
        Solution(energies, states, probs, degeneracy, psol.largest_discarded_probability, droplets, spins)
    end
    _merge
end


function merge_branches_blur(ctr::MpsContractor{T}, hamming_dist::Int, merge_type::Symbol=:nofit, update_droplets=NoDroplets()) where {T}
    function _merge_blur(psol::Solution)
        psol = merge_branches(ctr, merge_type, update_droplets)(psol)
        node = get(ctr.nodes_search_order, length(psol.states[1])+1, ctr.node_outside)
        boundaries = boundary_states(ctr, psol.states, node)
        # Get the indices that would sort the probabilities in descending order
        sorted_indices = sortperm(psol.probabilities, rev=true)
        # Use the sorted indices to reorder the boundary states and their probabilities
        sorted_boundaries = boundaries[sorted_indices]
        nsol = Solution(psol, Vector{Int}(sorted_indices)) #TODO Vector{Int} should be rm
        # Initialize an empty list for selected states
        selected_states = [] #Vector{Vector{Int}}()
        selected_idx = []
        # If sorted_boundaries is not empty, push the first state into selected_states
        if isempty(selected_states)
            push!(selected_states, sorted_boundaries[1])
            push!(selected_idx, 1)
        end

        for i in 2:length(sorted_boundaries)
            state = sorted_boundaries[i]

            # Check Hamming distance with existing selected states
            hamming_distances = [hamming_distance(state, s) for s in selected_states]
            # If no selected state has a Hamming distance greater than hamming_dist,
            # add the state to the selected_states list
            if all(hd >= hamming_dist for hd in hamming_distances)
                push!(selected_states, state)
                push!(selected_idx, i)
            end
        end
        Solution(nsol, Vector{Int}(selected_idx))
    end
    _merge_blur
end

hamming_distance(state1, state2) = state1 == state2 ? 0 : 1

"""
$(TYPEDSIGNATURES)
No-op merge function that returns the input `partial_sol` as is.

This function is a no-op merge function that takes a `Solution` object `partial_sol` as input and returns it unchanged. 
It is used as a merge strategy when you do not want to perform any merging of branches in a solution.

## Arguments
- `partial_sol::Solution`: A `Solution` object representing partial solutions.

## Returns
The input `partial_sol` object, unchanged.
"""
no_merge(partial_sol::Solution) = partial_sol

"""
$(TYPEDSIGNATURES)
Bound the solution to a specified number of states while discarding low-probability states.

This function takes a `Solution` object `psol`, bounds it to a specified number of states `max_states`, 
and discards low-probability states based on the probability threshold `δprob`. 
You can specify a `merge_strategy` for merging branches in the `psol` object.
    
## Arguments
- `psol::Solution`: A `Solution` object representing the solution to be bounded.
- `max_states::Int`: The maximum number of states to retain in the bounded solution.
- `δprob::Real`: The probability threshold for discarding low-probability states.
- `merge_strategy=no_merge`: (Optional) Merge strategy for branches. Defaults to `no_merge`.
    
## Returns
A `Solution` object representing the bounded solution with a maximum of `max_states` states.
    
"""
function bound_solution(
    psol::Solution, max_states::Int, δprob::Real, merge_strategy=no_merge
)
    psol = discard_probabilities!(merge_strategy(psol), δprob)
    if length(psol.probabilities) <= max_states return psol end
    idx = partialsortperm(psol.probabilities, 1:max_states + 1, rev=true)
    ldp = max(psol.largest_discarded_probability, psol.probabilities[idx[end]])
    Solution(psol, idx[1:max_states], ldp)
end

# """
# $(TYPEDSIGNATURES)
# """
function sampling(
    psol::Solution, max_states::Int, δprob::Real, merge_strategy=no_merge
)
    prob = exp.(psol.probabilities)
    new_prob = cumsum(reshape(prob, :, max_states), dims = 1)

    rr = rand(max_states)
    idx = zeros(max_states)
    idx_lin = zeros(Int, max_states)
    for (i, m) in enumerate(rr)
        np = new_prob[:, i]
        new_prob[:, i] = np / np[end]
        idx[i] = searchsortedfirst(new_prob[:, i], m)
        idx_lin[i] = Int((i-1) * size(new_prob, 1) + idx[i])
    end
    ldp = 0.0
    Solution(psol, idx_lin, ldp)
end

"""
$(TYPEDSIGNATURES)
Compute the low-energy spectrum of a spin glass PEPS network using branch-and-bound search.

This function computes the low-energy spectrum of a spin glass PEPS (Projected Entangled Pair State) network using a branch-and-bound search algorithm. 
It takes as input a `ctr` object representing the PEPS network and the parameters for controlling its contraction, `sparams` specifying search parameters, `merge_strategy` for merging branches, 
and `symmetry` indicating any symmetry constraints. Optionally, you can disable caching using the `no_cache` flag.

## Arguments
- `ctr::T`: The contractor object representing the PEPS network, which should be a subtype of `AbstractContractor`.
- `sparams::SearchParameters`: Parameters for controlling the search, including the maximum number of states and a cutoff probability.
- `merge_strategy=no_merge`: (Optional) Merge strategy for branches. Defaults to `no_merge`.
- `symmetry::Symbol=:noZ2`: (Optional) Symmetry constraint. Defaults to `:noZ2`.
- `no_cache=false`: (Optional) If `true`, disables caching. Defaults to `false`.

## Returns
A tuple `(sol, s)` containing:
- `sol::Solution`: A `Solution` object representing the computed low-energy spectrum.
- `s::Dict`: A dictionary containing Schmidt spectra for each row of the PEPS network.
"""
function low_energy_spectrum(
    ctr::T, sparams::SearchParameters, merge_strategy=no_merge, symmetry::Symbol=:noZ2; no_cache=false,
) where T <: AbstractContractor
    # Build all boundary mps
    CUDA.allowscalar(false)

    schmidts = Dict()
    @showprogress "Preprocessing: " for i ∈ ctr.peps.nrows + 1 : -1 : 2
        ψ0 = mps(ctr, i, length(ctr.betas))
        push!(schmidts, i=> measure_spectrum(ψ0))
        clear_memoize_cache_after_row()
        Memoization.empty_cache!(SpinGlassTensors.SparseCSC)
        empty!(ctr.peps.lp, :GPU)
        if i <= ctr.peps.nrows
            ψ0 = mps(ctr, i + 1, length(ctr.betas))
            move_to_CPU!(ψ0)
        end
    end

    s = Dict()
    for k in keys(schmidts)
        B = schmidts[k]
        v = []
        B = sort!(collect(B))
        for (i, _) in enumerate(B)
            push!(v, minimum(B[i][2]))
        end
        push!(s, k => v)
    end

    ψ0 = mps(ctr, 2, length(ctr.betas))
    move_to_CPU!(ψ0)

    # println("Memory memoize = ", measure_memory(Memoization.caches))
    # println("Memory lp = ", format_bytes.(measure_memory(ctr.peps.lp)), " elements = ", length(ctr.peps.lp))
    # println("Schmidt spectrum : remove those two lines and put it into sol")
    # println(schmidts)
    # Start branch and bound search
    sol = empty_solution()
    old_row = ctr.nodes_search_order[1][1]
    @showprogress "Search: " for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        current_row = node[1]
        if current_row > old_row
            old_row = current_row
            clear_memoize_cache_after_row()
            empty!(ctr.peps.lp, :GPU)
        end
        sol = branch_solution(sol, ctr)
        if symmetry == :Z2 && length(sol.states[1]) == 1
            indices_with_odd_numbers = Int[]
            for (index, vector) in enumerate(sol.spins)
                if any(isodd, vector)
                    push!(indices_with_odd_numbers, index)
                end
            end
            sol = Solution(sol, indices_with_odd_numbers)
        end
        sol = bound_solution(sol, sparams.max_states, sparams.cut_off_prob, merge_strategy)
        Memoization.empty_cache!(precompute_conditional)
        if no_cache Memoization.empty_all_caches!() end
    end
    clear_memoize_cache_after_row()
    empty!(ctr.peps.lp, :GPU)

    # Translate variable order (network --> factor graph)
    inner_perm = sortperm([
        ctr.peps.clustered_hamiltonian.reverse_label_map[idx]
        for idx ∈ ctr.peps.vertex_map.(ctr.nodes_search_order)
    ])
    
    inner_perm_inv = zeros(Int, length(inner_perm))
    inner_perm_inv[inner_perm] = collect(1:length(inner_perm))

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    sol = Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability,
        [perm_droplet(drop, inner_perm_inv) for drop in sol.droplets[outer_perm] ],
        sol.spins[outer_perm]
        # sol.pool_of_flips # TODO
    )

    # Final check if states correspond energies
    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.clustered_hamiltonian), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol, s
end

perm_droplet(drop::NoDroplets, perm::Vector{Int}) = drop

function perm_droplet(drops::Vector{Droplet}, perm::Vector{Int})
   [perm_droplet(drop, perm) for drop in drops]
end 

function perm_droplet(drop::Droplet, perm::Vector{Int})
    flip = drop.flip
    support = perm[flip.support]
    ind = sortperm(support)
    flip = Flip(support[ind], flip.state[ind], flip.spinxor[ind])
    Droplet(drop.denergy, drop.first, drop.last, flip, perm_droplet(drop.droplets, perm))
end

"""
$(TYPEDSIGNATURES)
Perform Gibbs sampling on a spin glass PEPS network.

This function performs Gibbs sampling on a spin glass PEPS (Projected Entangled Pair State) network using a branch-and-bound search algorithm. It takes as input a `ctr` object representing the PEPS network, `sparams` specifying search parameters, and `merge_strategy` for merging branches. Optionally, you can disable caching using the `no_cache` flag.

## Arguments

- `ctr::T`: The contractor object representing the PEPS network, which should be a subtype of `AbstractContractor`.
- `sparams::SearchParameters`: Parameters for controlling the search, including the maximum number of states and a cutoff probability.
- `merge_strategy=no_merge`: (Optional) Merge strategy for branches. Defaults to `no_merge`.
- `no_cache=false`: (Optional) If `true`, disables caching. Defaults to `false`.

## Returns

A `Solution` object representing the result of the Gibbs sampling.
"""
function gibbs_sampling(
    ctr::T, sparams::SearchParameters, merge_strategy=no_merge; no_cache=false,
) where T <: AbstractContractor
    # Build all boundary mps
    CUDA.allowscalar(false)

    @showprogress "Preprocessing: " for i ∈ ctr.peps.nrows:-1:1
        dressed_mps(ctr, i)
        clear_memoize_cache_after_row()
    end

    # Start branch and bound search
    sol = empty_solution(sparams.max_states)
    old_row = ctr.nodes_search_order[1][1]
    @showprogress "Search: " for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        current_row = node[1]
        if current_row > old_row
            old_row = current_row
            clear_memoize_cache_after_row()
        end
        sol = branch_solution(sol, ctr)
        sol = sampling(sol, sparams.max_states, sparams.cut_off_prob, merge_strategy)
        Memoization.empty_cache!(precompute_conditional)
        # TODO: clear memoize cache partially
        if no_cache Memoization.empty_all_caches!() end
    end
    clear_memoize_cache_after_row()

    # Translate variable order (network --> factor graph)
    inner_perm = sortperm([
        ctr.peps.clustered_hamiltonian.reverse_label_map[idx]
        for idx ∈ ctr.peps.vertex_map.(ctr.nodes_search_order)
    ])
    
    inner_perm_inv = zeros(Int, length(inner_perm))
    inner_perm_inv[inner_perm] = collect(1:length(inner_perm))

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    sol = Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability,
        [perm_droplet(drop, inner_perm_inv) for drop in sol.droplets[outer_perm]],
        sol.spins[outer_perm]
    )

    # Final check if states correspond energies
    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.clustered_hamiltonian), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol
end
