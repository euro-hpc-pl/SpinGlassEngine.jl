export
       SearchParameters,
       merge_branches,
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
       hamming_distance
"""
$(TYPEDSIGNATURES)
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

struct SingleLayerDroplets
    max_energy :: Real
    min_size :: Int
    metric :: Symbol 
    SingleLayerDroplets(max_energy = 1.0, min_size = 1, metric = :no_metric) = new(max_energy, min_size, metric)
end

struct Flip
    support :: Vector{Int}
    state :: Vector{Int}
    spinxor :: Vector{Int}
end

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
"""
@inline empty_solution(n::Int=1) = Solution(zeros(n), fill(Vector{Int}[], n), zeros(n), ones(Int, n),
                                            -Inf, repeat([NoDroplets()], n), fill(Vector{Int}[], n))
"""
$(TYPEDSIGNATURES)
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
"""
@inline function branch_energy(ctr::MpsContractor{T}, eσ::Tuple{<:Real, Vector{Int}}) where T
    eσ[begin] .+ update_energy(ctr, eσ[end])
end

"""
$(TYPEDSIGNATURES)
"""
@inline function branch_energies(ctr::MpsContractor{T}, psol::Solution) where T
    reduce(vcat, branch_energy.(Ref(ctr), zip(psol.energies, psol.states)))
end

"""
$(TYPEDSIGNATURES)
"""
function branch_states(local_basis::Vector{Int}, vec_states::Vector{Vector{Int}})
    states = reduce(hcat, vec_states)
    num_states = length(local_basis)
    lstate, nstates = size(states)
    ns = Array{Int}(undef, lstate+1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate+1, :, :] .= reshape(local_basis, num_states, 1, 1)
    collect(eachcol(reshape(ns, lstate+1, nstates * num_states)))
end

"""
$(TYPEDSIGNATURES)
"""
function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real, Vector{Int}}) where T
    pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))
end

"""
$(TYPEDSIGNATURES)
"""
function discard_probabilities!(psol::Solution, cut_off_prob::Real)
    pcut = maximum(psol.probabilities) + log(cut_off_prob)
    if minimum(psol.probabilities) >= pcut return psol end
    local_ldp = maximum(psol.probabilities[psol.probabilities .< pcut])
    ldp = max(local_ldp, psol.largest_discarded_probability)
    Solution(psol, findall(p -> p >= pcut, psol.probabilities), ldp)
end

"""
$(TYPEDSIGNATURES)
"""
function local_spins(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    spectrum(network, vertex).states_int
end

"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""

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

"""
$(TYPEDSIGNATURES)
"""
no_merge(partial_sol::Solution) = partial_sol

"""
$(TYPEDSIGNATURES)
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

"""
$(TYPEDSIGNATURES)
"""
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

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    sol = Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability,
        sol.droplets[outer_perm]# ,  # TODO add sort
        #sol.pool_of_flips
    )

    # Final check if states correspond energies
    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.clustered_hamiltonian), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol
end
