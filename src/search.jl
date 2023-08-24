export
       SearchParameters,
       merge_branches,
       low_energy_spectrum,
       Solution,
       bound_solution,
       gibbs_sampling,
       NoDroplets,
       SingleLayerDroplets,
       unpack_droplets,
       decode_to_spin
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
    SingleLayerDroplets(max_energy = 1.0, min_size = 1) = new(max_energy, min_size)
end

struct Flip
    support :: Vector{Int}
    state :: Vector{Int}
    spinxor :: Vector{Int}
end

mutable struct Droplet
    denergy :: Real  # excitation energy
    first :: Int  # site where droplet starts
    last :: Int  # site where droplet ends
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
end

"""
$(TYPEDSIGNATURES)
"""
@inline empty_solution(n::Int=1) = Solution(zeros(n), fill(Vector{Int}[], n), zeros(n), ones(Int, n), -Inf, repeat([NoDroplets()], n)) #, Dict())
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
        sol.droplets[idx]#,
        #sol.pool_of_flips
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
function branch_states(num_states::Int, vec_states::Vector{Vector{Int}})
    states = reduce(hcat, vec_states)
    lstate, nstates = size(states)
    local_basis = collect(1:num_states)
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
function branch_solution(psol::Solution, ctr::T) where T <: AbstractContractor
    num_states = cluster_size(ctr.peps, ctr.current_node)
    boundaries = boundary_states(ctr, psol.states, ctr.current_node)
    Solution(
        #reduce(vcat, branch_energy.(Ref(ctr), zip(psol.energies, psol.states))),
        branch_energies(ctr, psol),
        branch_states(num_states, psol.states),
        reduce(vcat, branch_probability.(Ref(ctr), zip(psol.probabilities, boundaries))),
        repeat(psol.degeneracy, inner=num_states),
        psol.largest_discarded_probability,
        repeat(psol.droplets, inner=num_states)#,
        #psol.pool_of_flips
    )
end

"""
$(TYPEDSIGNATURES)
"""

(method::NoDroplets)(ctr::MpsContractor{T}, best_idx::Int, energies::Vector{<:Real},  states::Vector{Vector{Int}}, droplets::Vector{Droplets}) where T= NoDroplets()

function (method::SingleLayerDroplets)(ctr::MpsContractor{T}, best_idx::Int, energies::Vector{<:Real}, states::Vector{Vector{Int}}, droplets::Vector{Droplets}) where T
    ndroplets = copy(droplets[best_idx])

    spins = decode_to_spin(ctr, states)

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
        ndroplets = my_push!(ndroplets, droplet, method)
        for subdroplet ∈ droplets[ind]
            new_droplet = merge_droplets(method, droplet, subdroplet)
            ndroplets = my_push!(ndroplets, new_droplet, method)
        end
    end
    ndroplets
end

function decode_to_spin(ctr::MpsContractor{T}, states::Vector{Vector{Int}}) where T
    fg = ctr.peps.factor_graph
    nodes = ctr.peps.vertex_map.(nodes_search_order_Mps(ctr.peps)[1])
    [[get_prop(fg, v, :spectrum).states_int[i] for (i, v) in zip(st, nodes)] for st in states]
end

function my_push!(ndroplets::Droplets, droplet::Droplet, method)
    if droplet.denergy <= method.max_energy && hamming_distance(droplet.flip) >= method.min_size
        if typeof(ndroplets) == NoDroplets
            ndroplets = Droplet[]
        end
        push!(ndroplets, droplet)
    end
    ndroplets
end

function hamming_distance(flip::Flip)
    sum(count_ones(st) for st ∈ flip.spinxor)
end

# function hamming_distance(flip1::Flip, flip2::Flip)
#     # TODO
#     # sum(count_ones(st) for st ∈ flip.spinxor)
# end


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

    for i in 1:length(sol.energies)
        push!(energies, sol.energies[i])
        push!(states, sol.states[i])
        push!(probs, sol.probabilities[i])
        push!(degeneracy, 1)
        push!(droplets, NoDroplets())

        for droplet in sol.droplets[i]
            push!(energies, sol.energies[i] + droplet.denergy)
            push!(states, flip_state(sol.states[i], droplet.flip))
            push!(probs, sol.probabilities[i] - β * droplet.denergy)
            push!(degeneracy, 1)
            push!(droplets, NoDroplets())
        end
    end

    inds = sortperm(energies)
    Solution(energies[inds], states[inds], probs[inds], degeneracy[inds], sol.largest_discarded_probability, droplets[inds])
end

# function update_droplets
# end

"""
$(TYPEDSIGNATURES)
"""
function merge_branches(ctr::MpsContractor{T}, merge_type::Symbol=:nofit, update_droplets=NoDroplets()) where {T}  # update_droplets=no_droplets
    function _merge(psol::Solution)
        node = get(ctr.nodes_search_order, length(psol.states[1])+1, ctr.node_outside)
        boundaries = boundary_states(ctr, psol.states, node)
        _, bnd_types = SpinGlassNetworks.unique_dims(boundaries, 1)

        sorting_idx = sortperm(bnd_types)
        sorted_bnd_types = bnd_types[sorting_idx]
        nsol = Solution(psol, Vector{Int}(sorting_idx)) #TODO Vector{Int} should be rm
        energies = typeof(nsol.energies[begin])[]
        states = typeof(nsol.states[begin])[]
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
            excitation = update_droplets(ctr, best_idx_bnd, nsol.energies[start:stop], nsol.states[start:stop], nsol.droplets[start:stop])
            push!(droplets, excitation)

            push!(energies, nsol.energies[best_idx])
            push!(states, nsol.states[best_idx])
            push!(degeneracy, new_degeneracy)
            start = stop + 1
        end
        Solution(energies, states, probs, degeneracy, psol.largest_discarded_probability, droplets) #, psol.pool_of_flips)
    end
    _merge
end



# """
# $(TYPEDSIGNATURES)
# """
# function merge_branches(ctr::MpsContractor{T}, merge_type::Symbol=:nofit) where {T}
#     function _merge(psol::Solution)
#         node = get(ctr.nodes_search_order, length(psol.states[1])+1, ctr.node_outside)
#         boundaries = boundary_states(ctr, psol.states, node)
#         _, bnd_types = SpinGlassNetworks.unique_dims(boundaries, 1)

#         sorting_idx = sortperm(bnd_types)
#         sorted_bnd_types = bnd_types[sorting_idx]
#         nsol = Solution(psol, Vector{Int}(sorting_idx)) #TODO Vector{Int} should be rm
#         energies = typeof(nsol.energies[begin])[]
#         states = typeof(nsol.states[begin])[]
#         probs = typeof(nsol.probabilities[begin])[]
#         degeneracy = typeof(nsol.degeneracy[begin])[]

#         start = 1
#         bsize = size(boundaries, 1)
#         while start <= bsize
#             stop = start
#             while stop + 1 <= bsize && sorted_bnd_types[start] == sorted_bnd_types[stop+1]
#                 stop = stop + 1
#             end
#             best_idx = argmin(@view nsol.energies[start:stop]) + start - 1

#             new_degeneracy = 0
#             ind_deg = []
#             for i in start:stop
#                 if nsol.energies[i] <= nsol.energies[best_idx] + 1E-12 # this is hack for now
#                     new_degeneracy += nsol.degeneracy[i]
#                     push!(ind_deg, i)
#                 end
#             end

#             if merge_type == :fit
#                 c = Statistics.median(
#                     ctr.betas[end] .* nsol.energies[start:stop] .+ nsol.probabilities[start:stop]
#                 )
#                 new_prob = -ctr.betas[end] .* nsol.energies[best_idx] .+ c
#                 push!(probs, new_prob)
#             elseif merge_type == :nofit
#                 push!(probs, nsol.probabilities[best_idx])
#             elseif merge_type == :python
#                 push!(probs, Statistics.mean(nsol.probabilities[ind_deg]))
#             end

#             push!(energies, nsol.energies[best_idx])
#             push!(states, nsol.states[best_idx])
#             push!(degeneracy, new_degeneracy)
#             start = stop + 1
#         end
#         Solution(energies, states, probs, degeneracy, psol.largest_discarded_probability)
#     end
#     _merge
# end
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
    ctr::T, sparams::SearchParameters, merge_strategy=no_merge; no_cache=false,
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
        sol = bound_solution(sol, sparams.max_states, sparams.cut_off_prob, merge_strategy)
        Memoization.empty_cache!(precompute_conditional)
        if no_cache Memoization.empty_all_caches!() end
    end
    clear_memoize_cache_after_row()
    empty!(ctr.peps.lp, :GPU)

    # Translate variable order (network --> factor graph)
    inner_perm = sortperm([
        ctr.peps.factor_graph.reverse_label_map[idx]
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
        sol.droplets[outer_perm] #,
        # sol.pool_of_flips # TODO
    )

    # Final check if states correspond energies
    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.factor_graph), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol, s
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
        ctr.peps.factor_graph.reverse_label_map[idx]
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
        Ref(ctr.peps.factor_graph), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol
end
