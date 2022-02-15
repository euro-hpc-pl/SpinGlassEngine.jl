export
       SVDTruncate,
       MPSAnnealing,
       MpoLayers,
       MpsParameters,
       MpsContractor,
       clear_memoize_cache,
       mps_top,
       mps,
       update_gauges!,
       boundary_state

abstract type AbstractContractor end
abstract type AbstractStrategy end

struct SVDTruncate <: AbstractStrategy end
struct MPSAnnealing <: AbstractStrategy end

"Distinguishes different layers of MPO that are used by the contraction algorithm."
struct MpoLayers
    main::Dict{Site, Sites}
    dress::Dict{Site, Sites}
    right::Dict{Site, Sites}
end

"Encapsulate control parameters for the MPO-MPS scheme used to contract the peps network."
struct MpsParameters
    bond_dimension::Int
    variational_tol::Real
    max_num_sweeps::Int

    MpsParameters(bd=typemax(Int), ϵ=1E-8, sw=4) = new(bd, ϵ, sw)
end

"Gives the layout used to construct the peps network."
layout(net::PEPSNetwork{T, S}) where {T, S} = T

"Gives the sparsity used to construct peps network."
sparsity(net::PEPSNetwork{T, S}) where {T, S} = S

"Tells how to contract the peps network using the MPO-MPS scheme."
mutable struct MpsContractor{T <: AbstractStrategy} <: AbstractContractor
    peps::PEPSNetwork{T, S} where {T, S}
    betas::Vector{<:Real}
    params::MpsParameters
    layers::MpoLayers
    statistics::Dict{Vector{Int}, <:Real}
    nodes_search_order::Vector{Node}
    node_search_index::Dict{Node, Int}
    current_node::Node

    function MpsContractor{T}(net, βs, params) where T
        ml = MpoLayers(layout(net), net.ncols)
        stat = Dict{Vector{Int}, Real}()
        ord = nodes_search_order_Mps(net)
        enum_ord = Dict(node => i for (i, node) ∈ enumerate(ord))
        node = ord[begin]
        new(net, βs, params, ml, stat, ord, enum_ord, node)
    end
end

"Gives the strategy to be used to contract peps network."
strategy(ctr::MpsContractor{T}) where {T} = T

"Construct (and memoize) MPO for a given layers."
@memoize Dict function mpo(
    ctr::MpsContractor{T}, layers::Dict{Site, Sites}, r::Int, indβ::Int
) where T <: AbstractStrategy
    mpo = Dict{Site, Dict{Site, Tensor}}()
    for (site, coordinates) ∈ layers
        lmpo = Dict{Site, Tensor}()
        for dr ∈ coordinates
            ten = tensor(ctr.peps, PEPSNode(r + dr, site), ctr.betas[indβ])
            push!(lmpo, dr => ten)
        end
        push!(mpo, site => lmpo)
    end
    QMpo(mpo)
end

"Construct (and memoize) top MPS using SVD for a given row."
@memoize Dict function mps_top(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    if i < 1
        W = mpo(ctr, ctr.layers.main, 1, indβ)
        return IdentityQMps(local_dims(W, :up))
    end

    ψ = mps_top(ctr, i-1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    ψ0 = dot(ψ, W)
    truncate!(ψ0, :left, ctr.params.bond_dimension)
    compress!(
        ψ0,
        W,
        ψ,
        ctr.params.bond_dimension,
        ctr.params.variational_tol,
        ctr.params.max_num_sweeps,
        :c
    )
    ψ0
end

"Construct (and memoize) (bottom) MPS using SVD for a given row."
@memoize Dict function mps(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(local_dims(W, :down))
    end

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    ψ0 = dot(W, ψ)
    truncate!(ψ0, :left, ctr.params.bond_dimension)
    compress!(
        ψ0,
        W,
        ψ,
        ctr.params.bond_dimension,
        ctr.params.variational_tol,
        ctr.params.max_num_sweeps
    )
    ψ0
end

"Construct (and memoize) (bottom) MPS using Annealing for a given row."
@memoize Dict function mps(ctr::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(local_dims(W, :down))
    end

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    if indβ > 1
        ψ0 = mps(ctr, i, indβ-1)
    else
        ψ0 = IdentityQMps(local_dims(W, :up), ctr.params.bond_dimension)
        canonise!(ψ0, :left)
    end

    compress!(
        ψ0,
        W,
        ψ,
        ctr.params.bond_dimension,
        ctr.params.variational_tol,
        ctr.params.max_num_sweeps
    )
    ψ0
end

"Construct dressed MPS for a given row and strategy."
function dressed_mps(ctr::MpsContractor{T}, i::Int) where T <: AbstractStrategy
    dressed_mps(ctr, i, length(ctr.betas))
end

"Construct (and memoize) dressed MPS for a given row and strategy."
@memoize Dict function dressed_mps(
    ctr::MpsContractor{T}, i::Int, indβ::Int
) where T <: AbstractStrategy
    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.dress, i, indβ)
    W * ψ
end

"Construct (and memoize) right environment for a given node."
@memoize Dict function right_env(
    ctr::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T <: AbstractStrategy
    l = length(∂v)
    if l == 0 return ones(1, 1) end

    R̃ = right_env(ctr, i, ∂v[2:l], indβ)
    ϕ = dressed_mps(ctr, i, indβ)
    W = mpo(ctr, ctr.layers.right, i, indβ)
    k = length(ϕ.sites)
    site = ϕ.sites[k-l+1]
    M = W[site]
    B = ϕ[site]

    RR = _update_reduced_env_right(R̃, ∂v[1], M, B)

    ls_mps = _left_nbrs_site(site, ϕ.sites)
    ls = _left_nbrs_site(site, W.sites)

    while ls > ls_mps
        M0 = W[ls][0]  # TODO: make this consistent
        @tensor RR[x, y] := M0[y, z] * RR[x, z]
        ls = _left_nbrs_site(ls, W.sites)
    end
    RR
end

function _update_reduced_env_right(
    RE::AbstractArray{Float64, 2}, m::Int, M::Dict, B::AbstractArray{Float64, 3}
)
    kk = sort(collect(keys(M)))
    if kk[1] < 0
        Mt = M[kk[1]]
        K = @view Mt[m, :]

        for ii ∈ kk[2:end]
            if ii == 0 break end
            Mm = M[ii]
            @tensor K[a] := K[b] * Mm[b, a]
        end
    else
        K = zeros(size(M[0], 2))
        K[m] = 1.
    end
    _update_reduced_env_right(K, RE, M[0], B)
end

function _update_reduced_env_right(
    K::AbstractArray{Float64, 1},
    RE::AbstractArray{Float64, 2},
    M::AbstractArray{Float64, 4},
    B::AbstractArray{Float64, 3}
)
    @tensor R[x, y] := K[d] * M[y, d, β, γ] * B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
    R
end

function _update_reduced_env_right(
    K::AbstractArray{Float64, 1},
    RE::AbstractArray{Float64, 2},
    M::SparseSiteTensor,
    B::AbstractArray{Float64, 3}
)
    @tensor REB[x, y, β] := B[x, y, α] * RE[α, β]

    Kloc_exp = M.loc_exp .* K[M.projs[2]]
    s3 = maximum(M.projs[4])
    ind43 = M.projs[4] .+ ((M.projs[3] .- 1) .* s3)
    @cast REB2[x, (y, z)] := REB[x, y, z]
    Rσ = REB2[:, ind43]

    R = zeros(size(B, 1), maximum(M.projs[1]))
    for (σ, kl) ∈ enumerate(Kloc_exp) R[:, M.projs[1][σ]] += kl .* Rσ[:, σ] end
    R
end

function _update_reduced_env_right(
    K::AbstractArray{Float64, 1},
    RE::AbstractArray{Float64, 2},
    M::SparseVirtualTensor,
    B::AbstractArray{Float64, 3}
)
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    @cast K2[t1, t2] := K[(t1, t2)] (t1 ∈ 1:maximum(p_rt))
    @tensor REB[x, y1, y2, β] := B4[x, y1, y2, α] * RE[α, β]
    R = zeros(size(B, 1), length(p_l))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        R[:, l] += (K2[p_rt[r], p_lt[l]] .* h[p_l[l], p_r[r]]) .* REB[:, p_lb[l], p_rb[r], r]
    end
    R
end

@memoize Dict function left_env(
    ctr::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T
    l = length(∂v)
    if l == 0 return ones(1) end
    L̃ = left_env(ctr, i, ∂v[1:l-1], indβ)
    ϕ = dressed_mps(ctr, i, indβ)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]
    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    L
end

function clear_memoize_cache()
    empty!(memoize_cache(left_env))
    empty!(memoize_cache(right_env))
    empty!(memoize_cache(mpo))
    empty!(memoize_cache(mps))
    empty!(memoize_cache(mps_top))
    empty!(memoize_cache(dressed_mps))
end

function error_measure(probs)
    if maximum(probs) <= 0 return 2.0 end
    if minimum(probs) < 0 return abs(minimum(probs)) / maximum(abs.(probs)) end
    return 0.0
end

function update_gauges!(
    ctr::MpsContractor{T},
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
    overlap = ψ_top * ψ_bot

    for i ∈ ψ_top.sites
        g = gauges[i]
        g_inv = 1.0 ./ g
        # Here we use convention that clm = ctr.layers.main with beginning and ending with matching gauges
        n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        n_top = PEPSNode(row + clm[i][end], i)
        g_top = ctr.peps.gauges.data[n_top] .* g
        g_bot = ctr.peps.gauges.data[n_bot] .* g_inv
        push!(ctr.peps.gauges.data, n_top => g_top, n_bot => g_bot)
    end

    for ind ∈ 1:indβ
        for i ∈ row:ctr.peps.nrows delete!(memoize_cache(mps_top), (ctr, i, ind)) end
        for i ∈ 1:row+1 delete!(memoize_cache(mps), (ctr, i, ind)) end
        for i ∈ row:row+1
            delete!(memoize_cache(mpo), (ctr, ctr.layers.main, i, ind))
            delete!(memoize_cache(mpo), (ctr, ctr.layers.dress, i, ind))
            delete!(memoize_cache(mpo), (ctr, ctr.layers.right, i, ind))
        end
    end
    overlap
end

function conditional_probability(ctr::MpsContractor{S}, w::Vector{Int}) where S
    conditional_probability(layout(ctr.peps), ctr, w)
end

function update_energy(ctr::MpsContractor{S}, w::Vector{Int}) where S
    update_energy(layout(ctr.peps), ctr, w)
end

function boundary_state(
    ctr::MpsContractor{T}, σ::Vector{Int}, node::S
) where {T, S}
    boundary_index.(Ref(ctr), boundary(ctr, node), Ref(σ))
end

function boundary(ctr::MpsContractor{T}, node::Node) where T
    boundary(layout(ctr.peps), ctr, node)
end

function boundary_index(
    ctr::MpsContractor{T},
    nodes::Tuple{S, Union{S, NTuple{N, S}}},
    σ::Vector{Int}
) where {S, T, N}
    v, w = nodes
    state = local_state_for_node(ctr, σ, v)
    if ctr.peps.vertex_map(v) ∉ vertices(ctr.peps.factor_graph) return ones_like(state) end
    projector(ctr.peps, v, w)[state]
end

""" boundary index formed from outer product of two projectors"""
function boundary_index(
    ctr::MpsContractor{T}, nodes::NTuple{4, S}, σ::Vector{Int}
) where {S, T}
    v, w, k, l = nodes
    pv = projector(ctr.peps, v, w)
    i = boundary_index(ctr, (v, w), σ)
    j = boundary_index(ctr, (k, l), σ)
    (j - 1) * maximum(pv) + i
end

function local_state_for_node(
    ctr::MpsContractor{T}, σ::Vector{Int}, w::S
) where {T, S}
    k = get(ctr.node_search_index, w, 0)
    0 < k <= length(σ) ? σ[k] : 1
    # 0 < k ? σ[k] : 1  # likely we shouldnt be asking for node with k > length(σ) -- but this does not work
end
