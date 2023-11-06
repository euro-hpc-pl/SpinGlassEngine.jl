# droplets.jl: This file provides functions for finding droplets and operating on them.

export
    NoDroplets,
    SingleLayerDroplets,
    Flip,
    Droplet,
    Droplets,
    hamming_distance,
    unpack_droplets,
    perm_droplet

struct NoDroplets end

Base.iterate(drop::NoDroplets) = nothing
Base.copy(s::NoDroplets) = s
Base.getindex(s::NoDroplets, ::Any) = NoDroplets()

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
    max_energy::Real
    min_size::Int
    metric::Symbol

    SingleLayerDroplets(
        max_energy = 1.0,
        min_size = 1,
        metric = :no_metric
    ) = new(max_energy, min_size, metric)
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
    support::Vector{Int}
    state::Vector{Int}
    spinxor::Vector{Int}
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
    denergy::Real  # excitation energy
    first::Int  # site where droplet starts
    last::Int
    flip::Flip  # states of nodes flipped by droplet
    droplets::Union{NoDroplets, Vector{Droplet}}  # subdroplets on top of the current droplet; can be empty
end

Droplets = Union{NoDroplets, Vector{Droplet}}  # Can't be defined before Droplet struct

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
) where T = NoDroplets()

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

hamming_distance(flip::Flip) = sum(count_ones(st) for st ∈ flip.spinxor)
hamming_distance(state1, state2) = sum(state1 .!== state2)

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