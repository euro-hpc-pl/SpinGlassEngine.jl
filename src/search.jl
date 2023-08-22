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
       SingleLayerDropletsHamming,
       unpack_droplets_hamming
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


struct SingleLayerDroplets
    max_energy :: Real
    min_size :: Int
    SingleLayerDroplets(max_energy = 1.0, min_size = 1) = new(max_energy, min_size)
end

struct SingleLayerDropletsHamming
    max_energy :: Real
    min_size :: Int
    SingleLayerDropletsHamming(max_energy = 1.0, min_size = 1) = new(max_energy, min_size)
end

mutable struct Droplet
    denergy :: Real  # excitation energy
    from :: Int  # site where droplet starts
    to :: Int  # site where droplet ends
    flip :: Vector{Int}  # Int  # switch to key in pool_of_flips ? or can we have some immutable type
    droplets :: Union{NoDroplets, Vector{Droplet}}  # droplets on top of the current droplet; can be empty
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
    # pool_of_flips::Dict
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

(method::NoDroplets)(best_idx::Int, energies::Vector{<:Real},  states::Vector{Vector{Int}}, droplets::Vector{Droplets}) = NoDroplets()

function (method::SingleLayerDroplets)(best_idx::Int, energies::Vector{<:Real}, states::Vector{Vector{Int}}, droplets::Vector{Droplets})
    ndroplets = deepcopy(droplets[best_idx])

    bstate  = states[best_idx]
    benergy = energies[best_idx]

    for ind ∈ (1 : best_idx - 1..., best_idx + 1 : length(energies)...)
        flip = xor.(bstate, states[ind])
        first = findfirst(x -> x != 0, flip)
        last = findlast(x -> x != 0, flip)
        flip = flip[first:last]
        deng = energies[ind] - benergy

        droplet = Droplet(deng, first, last, flip, NoDroplets())
        ndroplets = my_push!(ndroplets, droplet, deng, method)
        
        if typeof(droplets[ind]) != NoDroplets
            for subdroplet ∈ droplets[ind]
                new_droplet = droplet_xor_simple(droplet, subdroplet)
                ndroplets = my_push!(ndroplets, new_droplet, new_droplet.denergy, method)
            end
        end
    end
    ndroplets
end

function (method::SingleLayerDropletsHamming)(best_idx::Int, energies::Vector{<:Real}, states::Vector{Vector{Int}}, droplets::Vector{Droplets})
    ndroplets = deepcopy(droplets[best_idx])

    bstate  = states[best_idx]
    benergy = energies[best_idx]

    for ind ∈ (1 : best_idx - 1..., best_idx + 1 : length(energies)...)
        flip = states[ind] #xor.(bstate, states[ind])
        first = findall(bstate.!=states[ind])[begin] #findfirst(x -> x != 0, flip)
        last = findall(bstate.!=states[ind])[end] #findlast(x -> x != 0, flip)
        flip = flip[first:last]
        deng = energies[ind] - benergy
        droplet = Droplet(deng, first, last, flip, NoDroplets())
        ndroplets = my_push!(ndroplets, droplet, deng, method)
        
        if typeof(droplets[ind]) != NoDroplets
            for subdroplet ∈ droplets[ind]
                new_droplet = droplet_hamming_simple(droplet, subdroplet)
                ndroplets = my_push!(ndroplets, new_droplet, new_droplet.denergy, method)
            end
        end
    end
    ndroplets
end


function my_push!(ndroplets::Droplets, droplet::Droplet, deng::Real, method)

    if deng <= method.max_energy #&& hamming_distance(ndroplets.flip, droplet.flip) < method.min_size
        if typeof(ndroplets) == NoDroplets
            ndroplets = Droplet[]
        end
        push!(ndroplets, droplet)
    end
    ndroplets
end

function hamming_distance(s1::Vector{Int}, s2::Vector{Int})
    if length(s1) != length(s2)
        throw(ArgumentError("Undefined for sequences of unequal length."))
    end
    
    distance = sum(el1 != el2 for (el1, el2) in zip(s1, s2))
    return distance
end

function droplet_xor_simple(drop1, drop2)
    denergy = drop1.denergy + drop2.denergy
    first = min(drop1.from, drop2.from)
    last = max(drop1.to, drop2.to)
    flip = zeros(Int, last - first + 1)
    flip[drop1.from - first + 1 : drop1.to - last + 1] .⊻= drop1.flip
    flip[drop2.from - first + 1 : drop2.to - last + 1] .⊻= drop2.flip
    Droplet(denergy, first, last, flip, NoDroplets())
end

function droplet_hamming_simple(drop1, drop2)
    denergy = drop1.denergy + drop2.denergy
    first = min(drop1.from, drop2.from)
    last = max(drop1.to, drop2.to)
    flip = zeros(last - first + 1)
     for i in 1:length(flip)
         idx_drop1 = first - drop1.from + i
         idx_drop2 = first - drop2.from + i
         if 1 <= idx_drop1 <= length(drop1.flip) && 1 <= idx_drop2 <= length(drop2.flip)
             flip[i] = drop2.flip[idx_drop2]
         elseif 1 <= idx_drop1 <= length(drop1.flip)
             flip[i] = drop1.flip[idx_drop1]
         elseif 1 <= idx_drop2 <= length(drop2.flip)
             flip[i] = drop2.flip[idx_drop2]
         else
             flip[i] = 0  # Default value if both flips are out of bounds
         end
     end
    Droplet(denergy, first, last, flip, NoDroplets())
end

function flip_state(state::Vector{Int}, droplet::Droplet)
    new_state = copy(state)
    new_state[droplet.from:droplet.to] .⊻= droplet.flip
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
            push!(states, flip_state(sol.states[i], droplet))
            push!(probs, sol.probabilities[i] - β * droplet.denergy)
            push!(degeneracy, 1)
            push!(droplets, NoDroplets())
        end
    end

    inds = sortperm(energies)
    Solution(energies[inds], states[inds], probs[inds], degeneracy[inds], sol.largest_discarded_probability, droplets[inds])
end

function flip_state2(state::Vector{Int}, droplet::Droplet)
    new_state = copy(state)
    new_state[droplet.from:droplet.to] .= droplet.flip
    new_state
end

function unpack_droplets_hamming(sol, β)  # have β in sol ?
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
            push!(states, flip_state2(sol.states[i], droplet))
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
            excitation = update_droplets(best_idx_bnd, nsol.energies[start:stop], nsol.states[start:stop], nsol.droplets[start:stop])
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

