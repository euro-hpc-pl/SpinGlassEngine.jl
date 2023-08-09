export
       SearchParameters,
       merge_branches,
       low_energy_spectrum,
       Solution,
       bound_solution,
       gibbs_sampling

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


# struct NoDroplet end

# mutable struct NonemptyDroplet
#     denergy :: Real  # excitation energy
#     from :: Int  # site where droplet starts
#     to :: Int  # site where droplet ends
#     flip :: Int  # key to pool_of_flips
#     droplets :: Vector{Droplet}  # droplets on top of the current droplet; can be empty
# end

# Droplet = Union{}

"""
$(TYPEDSIGNATURES)
"""
struct Solution
    energies::Vector{<:Real}
    states::Vector{Vector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
    # excitations::Vector{Vector{Droplet}}  # no droplets = Vector{Empty Vectors}
    # pool_of_flips::Dict
end

"""
$(TYPEDSIGNATURES)
"""
@inline empty_solution(n::Int=1) = Solution(zeros(n), fill(Vector{Int}[], n) , zeros(n), ones(Int, n), -Inf)
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
        ldp
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
        psol.largest_discarded_probability
    )
end

"""
$(TYPEDSIGNATURES)
"""
function merge_branches(ctr::MpsContractor{T}, merge_type::Symbol=:nofit) where {T}  # update_droplets=no_droplets
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

        start = 1
        bsize = size(boundaries, 1)
        while start <= bsize
            stop = start
            while stop + 1 <= bsize && sorted_bnd_types[start] == sorted_bnd_types[stop+1]
                stop = stop + 1
            end
            best_idx = argmin(@view nsol.energies[start:stop]) + start - 1

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
            # excitation = update_excitations(best_idx, start, stop, states, energies, droplets; parameters_which_droplets_to_keep)
            # push!(droplets, excitation)

            push!(energies, nsol.energies[best_idx])
            push!(states, nsol.states[best_idx])
            push!(degeneracy, new_degeneracy)
            start = stop + 1
        end
        Solution(energies, states, probs, degeneracy, psol.largest_discarded_probability)
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
        sol.largest_discarded_probability
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
        sol.largest_discarded_probability
    )

    # Final check if states correspond energies
    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.factor_graph), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol
end

