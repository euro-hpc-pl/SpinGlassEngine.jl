export
       SVDTruncate,
       MPSAnnealing,
       Zipper,
       MpoLayers,
       MpsParameters,
       MpsContractor,
       NoUpdate,
       GaugeStrategy,
       GaugeStrategyWithBalancing,
       clear_memoize_cache,
       mps_top,
       mps,
       mps_top_approx,
       mps_approx,
       update_gauges!,
       update_gauges_with_balancing!,
       boundary_states

abstract type AbstractContractor end
abstract type AbstractStrategy end
abstract type AbstractGauge end

struct SVDTruncate <: AbstractStrategy end
struct MPSAnnealing <: AbstractStrategy end
struct Zipper <: AbstractStrategy end
struct GaugeStrategyWithBalancing <: AbstractGauge end
struct GaugeStrategy <: AbstractGauge end
struct NoUpdate <: AbstractGauge end

"""
$(TYPEDSIGNATURES)

Distinguishes different layers of MPO that are used by the contraction algorithm.
"""
struct MpoLayers
    main::Dict{Site, Sites}
    dress::Dict{Site, Sites}
    right::Dict{Site, Sites}
end

"""
$(TYPEDSIGNATURES)

Encapsulate control parameters for the MPO-MPS scheme used to contract the peps network.
"""
struct MpsParameters
    bond_dimension::Int
    variational_tol::Real
    max_num_sweeps::Int
    tol_SVD::Real

    MpsParameters(bd=typemax(Int), ϵ=1E-8, sw=4, ts=1E-16) = new(bd, ϵ, sw, ts)
end

"""
$(TYPEDSIGNATURES)

Gives the layout used to construct the peps network.
"""
layout(net::PEPSNetwork{T, S}) where {T, S} = T

"""
$(TYPEDSIGNATURES)

Gives the sparsity used to construct peps network.
"""
sparsity(net::PEPSNetwork{T, S}) where {T, S} = S

"""
$(TYPEDSIGNATURES)

Tells how to contract the peps network using the MPO-MPS scheme.
"""
mutable struct MpsContractor{T <: AbstractStrategy, R <: AbstractGauge} <: AbstractContractor
    peps::PEPSNetwork{T, S} where {T, S}
    betas::Vector{<:Real}
    graduate_truncation::Symbol
    params::MpsParameters
    layers::MpoLayers
    statistics#::Dict{Vector{Int}, <:Real}
    nodes_search_order::Vector{Node}
    node_outside::Node
    node_search_index::Dict{Node, Int}
    current_node::Node

    function MpsContractor{T, R}(net, βs, graduate_truncation::Symbol, params) where {T, R}
        ml = MpoLayers(layout(net), net.ncols)
        stat = Dict()
        ord, node_out = nodes_search_order_Mps(net)
        enum_ord = Dict(node => i for (i, node) ∈ enumerate(ord))
        node = ord[begin]
        new(net, βs, graduate_truncation, params, ml, stat, ord, node_out, enum_ord, node)
    end
end

"""
$(TYPEDSIGNATURES)

Gives the strategy to be used to contract peps network.
"""
strategy(ctr::MpsContractor{T}) where {T} = T

"""
$(TYPEDSIGNATURES)

Construct (and memoize) MPO for a given layers.
"""
@memoize Dict function mpo(
    ctr::MpsContractor{T}, layers::Dict{Site, Sites}, r::Int, indβ::Int
) where T <: AbstractStrategy
    mpo = Dict{Site, MpoTensor{Float64}}() # Float64 - for now
    for (site, coordinates) ∈ layers
        lmpo = TensorMap{Float64}()  # Float64 - for now
        for dr ∈ coordinates
            ten = tensor(ctr.peps, PEPSNode(r + dr, site), ctr.betas[indβ])
            push!(lmpo, dr => ten)
        end
        push!(mpo, site => MpoTensor(lmpo))
    end
    move_to_CUDA!(QMpo(mpo))
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) top MPS using SVD for a given row.
"""
@memoize Dict function mps_top(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i < 1
        W = mpo(ctr, ctr.layers.main, 1, indβ)
        return IdentityQMps(Float64, local_dims(W, :up); onGPU=true) # F64 for now
    end

    ψ = mps_top(ctr, i-1, indβ)
    W = transpose(mpo(ctr, ctr.layers.main, i, indβ))
    ψ0 = dot(W, ψ)

    canonise!(ψ0, :right)
    if ctr.graduate_truncation == :graduate_truncate
        canonise_truncate!(ψ0, :left, Dcut * 2, tolS / 2)
        variational_sweep!(ψ0, W, ψ, Val(:right))
    end
    canonise_truncate!(ψ0, :left, Dcut, tolS)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) (bottom) MPS using SVD for a given row.
"""
@memoize Dict function mps(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(Float64, local_dims(W, :down); onGPU=true) # Float64 fror now
    end

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    ψ0 = dot(W, ψ)
    canonise!(ψ0, :right)
    if ctr.graduate_truncation == :graduate_truncate
        canonise_truncate!(ψ0, :left, Dcut * 2, tolS / 2)
        variational_sweep!(ψ0, W, ψ, Val(:right))
    end
    canonise_truncate!(ψ0, :left, Dcut, tolS)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) top MPS using SVD for a given row.
"""
# @memoize Dict function mps_top_approx(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
#     if i < 1
#         W = mpo(ctr, ctr.layers.main, 1, indβ)
#         return IdentityQMps(local_dims(W, :up))
#     end

#     W = mpo(ctr, ctr.layers.main, i, indβ)
#     ψ = IdentityQMps(local_dims(W, :up))

#     ψ0 = dot(ψ, W)
#     truncate!(ψ0, :left, ctr.params.bond_dimension)
#     ψ0
# end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) (bottom) MPS using SVD for a given row.
"""
@memoize Dict function mps_approx(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(Float64, local_dims(W, :down); onGPU=true) # F64 for now
    end

    W = mpo(ctr, ctr.layers.main, i, indβ)
    ψ = IdentityQMps(Float64, local_dims(W, :down); onGPU=true) # F64 for now

    ψ0 = dot(W, ψ)
    truncate!(ψ0, :left, ctr.params.bond_dimension)
    ψ0
end

#=
function (ctr::MpsContractor)(peps::PEPSNetwork, ...., :mps_top)

for ctr ∈ [ctr_1, ctr_2]
    mpo_1 = ctr_1(peps, ..., :mpo)
    mpo_2 = ctr_2(peps, ..., :mpo)

    #@nexprs 2 k k -> mpo_k = ctr_k(peps, ..., :mpo)
end
=#

"""
$(TYPEDSIGNATURES)

Construct (and memoize) top MPS using Zipper (truncated SVD) for a given row.
"""
@memoize Dict function mps_top(ctr::MpsContractor{Zipper}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i < 1
        W = mpo(ctr, ctr.layers.main, 1, indβ)
        return IdentityQMps(Float64, local_dims(W, :up)) # F64 for now
    end

    ψ = mps_top(ctr, i-1, indβ)
    W = transpose(mpo(ctr, ctr.layers.main, i, indβ))

    canonise!(ψ, :left)
    ψ0 = zipper(W, ψ, method=:psvd_sparse, Dcut=Dcut, tol=tolS)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) (bottom) MPS using Zipper (truncated SVD) for a given row.
"""
@memoize Dict function mps(ctr::MpsContractor{Zipper}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(Float64, local_dims(W, :down)) # Float64 fror now
    end

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    canonise!(ψ, :left)
    ψ0 = zipper(W, ψ, method=:psvd_sparse, Dcut=Dcut, tol=tolS)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) (bottom) top MPS using Annealing for a given row.
"""
@memoize Dict function mps_top(ctr::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
    if i < 1
        W = mpo(ctr, ctr.layers.main, 1, indβ)
        return IdentityQMps(Float64, local_dims(W, :up)) # F64 for now
    end

    ψ = mps_top(ctr, i-1, indβ)
    W = transpose(mpo(ctr, ctr.layers.main, i, indβ))

    if indβ > 1
        ψ0 = mps_top(ctr, i, indβ-1)
    else
        ψ0 = IdentityQMps(Float64, local_dims(W, :up), ctr.params.bond_dimension; onGPU=true) # F64 for now
        # ψ0 = IdentityQMps(Float64, local_dims(W, :down), ctr.params.bond_dimension) # F64 for now
        canonise!(ψ0, :left)
    end
    variational_compress!(
        ψ0,
        W,
        ψ,
        ctr.params.variational_tol,
        ctr.params.max_num_sweeps,
    )
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) (bottom) MPS using Annealing for a given row.
"""
@memoize Dict function mps(ctr::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(Float64, local_dims(W, :down)) # F64 for now
    end

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    if indβ > 1
        ψ0 = mps(ctr, i, indβ-1)
    else
        ψ0 = IdentityQMps(Float64, local_dims(W, :up), ctr.params.bond_dimension; onGPU=true) # F64 for now
        canonise!(ψ0, :left)
    end

    variational_compress!(
        ψ0,
        W,
        ψ,
        ctr.params.variational_tol,
        ctr.params.max_num_sweeps
    )
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct dressed MPS for a given row and strategy.
"""
function dressed_mps(ctr::MpsContractor{T}, i::Int) where T <: AbstractStrategy
    dressed_mps(ctr, i, length(ctr.betas))
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) dressed MPS for a given row and strategy.
"""
@memoize Dict function dressed_mps(
    ctr::MpsContractor{T}, i::Int, indβ::Int
) where T <: AbstractStrategy

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.dress, i, indβ)
    ϕ = dot(W, ψ)

    for j ∈ ϕ.sites
        nrm = maximum(abs.(ϕ[j]))
        if !iszero(nrm) ϕ[j] ./= nrm end
    end
    ϕ
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) right environment for a given node.
"""
@memoize Dict function right_env(
    ctr::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T <: AbstractStrategy
    l = length(∂v)
    if l == 0 return CUDA.ones(Float64, 1, 1) end  # TODO type propagation

    R̃ = right_env(ctr, i, ∂v[2:l], indβ)
    ϕ = dressed_mps(ctr, i, indβ)
    W = mpo(ctr, ctr.layers.right, i, indβ)  # WRONG FLOW; now mpo is not memoised
    k = length(ϕ.sites)
    site = ϕ.sites[k-l+1]
    M = W[site]
    B = ϕ[site]

    RR = update_reduced_env_right(R̃, ∂v[1], M, B)

    ls_mps = left_nbrs_site(site, ϕ.sites)
    ls = left_nbrs_site(site, W.sites)

    while ls > ls_mps
        RR = update_reduced_env_right(RR, W[ls].ctr)
        ls = left_nbrs_site(ls, W.sites)
    end
    nmr = maximum(abs.(RR))
    iszero(nmr) ? RR : RR ./ nmr
end


"""
$(TYPEDSIGNATURES)
"""
@memoize Dict function left_env(
    ctr::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T
    l = length(∂v)
    if l == 0 return CUDA.ones(Float64, 1) end
    L̃ = left_env(ctr, i, ∂v[1:l-1], indβ)
    ϕ = dressed_mps(ctr, i, indβ)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]

    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    nmr = maximum(abs.(L))
    iszero(nmr) ? L : L ./ nmr
end

"""
$(TYPEDSIGNATURES)
"""
function clear_memoize_cache()
    Memoization.empty_all_caches!()
end

function clear_memoize_cache_after_row()
    Memoization.empty_cache!.((left_env, right_env, mpo))
end

"""
$(TYPEDSIGNATURES)
"""
function clear_memoize_cache(ctr::MpsContractor{T, S}, row::Site, indβ::Int) where {T, S}
    for ind ∈ 1:indβ # indbeta a vector?
        for i ∈ row:ctr.peps.nrows
            delete!(Memoization.caches[mps_top], ((ctr, i, ind), ()))
        end
        for i ∈ 1:row+1
            delete!(Memoization.caches[mps], ((ctr, i, ind), ()))
        end
        for i ∈ row:row+2
            cmpo = Memoization.caches[mpo]
            delete!(cmpo, ((ctr, ctr.layers.main, i, ind), ()))
            delete!(cmpo, ((ctr, ctr.layers.dress, i, ind), ()))
            delete!(cmpo, ((ctr, ctr.layers.right, i, ind), ()))
        end

    end
end

"""
$(TYPEDSIGNATURES)
"""
function error_measure(probs)
    if maximum(probs) <= 0 return 2.0 end
    if minimum(probs) < 0 return abs(minimum(probs)) / maximum(abs.(probs)) end
    return 0.0
end

"""
$(TYPEDSIGNATURES)
"""
function sweep_gauges!(
    ctr::MpsContractor{T, GaugeStrategy},
    row::Site,
    indβ::Int,
    tol::Real=1E-4,
    max_sweeps::Int=10
) where T
    clm = ctr.layers.main
    ψ_top = mps_top(ctr, row, indβ)
    ψ_bot = mps(ctr, row + 1, indβ)

    ψ_top = deepcopy(ψ_top)
    ψ_bot = deepcopy(ψ_bot)

    gauges = optimize_gauges_for_overlaps!!(ψ_top, ψ_bot, tol, max_sweeps)

    for i ∈ ψ_top.sites
        g = gauges[i]
        g_inv = 1.0 ./ g
        @inbounds n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        @inbounds n_top = PEPSNode(row + clm[i][end], i)
        g_top = ctr.peps.gauges.data[n_top] .* g
        g_bot = ctr.peps.gauges.data[n_bot] .* g_inv
        push!(ctr.peps.gauges.data, n_top => g_top, n_bot => g_bot)
    end
    clear_memoize_cache(ctr, row, indβ)
end

"""
$(TYPEDSIGNATURES)
"""
function sweep_gauges!(
    ctr::MpsContractor{T, GaugeStrategyWithBalancing},
    row::Site,
    indβ::Int,
    ) where T
    clm = ctr.layers.main
    ψ_top = mps_top(ctr, row, indβ)
    ψ_bot = mps(ctr, row + 1, indβ)
    ψ_top = deepcopy(ψ_top)
    ψ_bot = deepcopy(ψ_bot)
    for i ∈ ψ_top.sites
        @inbounds n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        @inbounds n_top = PEPSNode(row + clm[i][end], i)
        ρ = overlap_density_matrix(ψ_top, ψ_bot, i)
        _, _, scale = LinearAlgebra.LAPACK.gebal!('S', ρ)
        push!(ctr.peps.gauges.data, n_top => 1.0 ./ scale, n_bot => scale)
    end
    clear_memoize_cache(ctr, row, indβ)
    ψ_top * ψ_bot
end

"""
$(TYPEDSIGNATURES)
"""
function sweep_gauges!(
    ctr::MpsContractor{T, NoUpdate},
    row::Site,
    indβ::Int,
    tol::Real=1E-4,
    max_sweeps::Int=10
) where T

end

"""
$(TYPEDSIGNATURES)
"""
function update_gauges!(
    ctr::MpsContractor{T, S},
    row::Site,
    indβ::Vector{Int},
    ::Val{:down}
) where {T, S}
    for j ∈ indβ, i ∈ 1:row-1
        sweep_gauges!(ctr, i, j)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function update_gauges!(
    ctr::MpsContractor{T, S},
    row::Site,
    indβ::Vector{Int},
    ::Val{:up}
) where {T, S}
    for j ∈ indβ, i ∈ row-1:-1:1
        sweep_gauges!(ctr, i, j)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function conditional_probability(ctr::MpsContractor{S}, w::Vector{Int}) where S
    conditional_probability(layout(ctr.peps), ctr, w)
end

"""
$(TYPEDSIGNATURES)
"""
function update_energy(ctr::MpsContractor{S}, w::Vector{Int}) where S
    update_energy(layout(ctr.peps), ctr, w)
end

function boundary_states(
    ctr::MpsContractor{T}, states::Vector{Vector{Int}}, node::S
) where {T, S}
    boundary_recipe = boundary(ctr, node)
    res = ones(Int, length(states), length(boundary_recipe))
    for (i, node) ∈ enumerate(boundary_recipe)
        @inbounds res[:, i] = boundary_indices(ctr, node, states)
    end
    [res[r, :] for r ∈ 1:size(res, 1)]
end

"""
$(TYPEDSIGNATURES)
"""
function boundary(ctr::MpsContractor{T}, node::Node) where T
    boundary(layout(ctr.peps), ctr, node)
end

"""
$(TYPEDSIGNATURES)
"""
function local_state_for_node(
    ctr::MpsContractor{T}, σ::Vector{Int}, w::S
) where {T, S}
    k = get(ctr.node_search_index, w, 0)
    0 < k <= length(σ) ? σ[k] : 1
end

"""
$(TYPEDSIGNATURES)
"""
function boundary_indices(
    ctr::MpsContractor{T},
    nodes::Union{NTuple{2, S}, Tuple{S, NTuple{N, S}}},
    states::Vector{Vector{Int}}
) where {T, S, N}
    v, w = nodes
    if ctr.peps.vertex_map(v) ∈ vertices(ctr.peps.factor_graph)
        @inbounds idx = [σ[ctr.node_search_index[v]] for σ ∈ states]
        return @inbounds projector(ctr.peps, v, w)[idx]
    end
    ones(Int, length(states))
end

"""
$(TYPEDSIGNATURES)

boundary index formed from outer product of two projectors
"""
function boundary_indices(
    ctr::MpsContractor{T}, nodes::Union{NTuple{4, S}, Tuple{S, NTuple{2, S}, S, NTuple{2, S}}}, states::Vector{Vector{Int}}
) where {S, T}
    v, w, k, l = nodes
    pv = projector(ctr.peps, v, w)
    i = boundary_indices(ctr, (v, w), states)
    j = boundary_indices(ctr, (k, l), states)
    (j .- 1) .* maximum(pv) .+ i
end


function boundary_indices(
    ctr::MpsContractor{T}, nodes::Tuple{S, NTuple{2, S}, S, NTuple{2, S}, S, NTuple{2, S}, S, NTuple{2, S}}, states::Vector{Vector{Int}}
) where {S, T}
    v1, v2, v3, v4, v5, v6, v7, v8 = nodes
    pv1 = projector(ctr.peps, v1, v2)
    pv3 = projector(ctr.peps, v3, v4)
    mm = maximum(pv1) * maximum(pv3)
    i = boundary_indices(ctr, (v1, v2, v3, v4), states)
    j = boundary_indices(ctr, (v5, v6, v7, v8), states)
    (j .- 1) .* mm .+ i
end
