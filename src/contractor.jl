export SVDTruncate, MPSAnnealing, MpoLayers, MpsParameters, MpsContractor
export clear_memoize_cache, mps_top, mps, update_gauges!

abstract type AbstractContractor end
abstract type AbstractStrategy end

struct SVDTruncate <: AbstractStrategy end
struct MPSAnnealing <: AbstractStrategy end

struct MpoLayers
    main::Dict{Site, Sites}
    dress::Dict{Site, Sites}
    right::Dict{Site, Sites}
end

MpoLayers
struct MpsParameters
    bond_dimension::Int
    variational_tol::Real
    max_num_sweeps::Int

    MpsParameters(bd=typemax(Int), ϵ=1E-8, sw=4) = new(bd, ϵ, sw)
end
layout(net::PEPSNetwork{T, S}) where {T, S} = T
sparsity(net::PEPSNetwork{T, S}) where {T, S} = S
mutable struct MpsContractor{T <: AbstractStrategy} <: AbstractContractor
    peps::PEPSNetwork{T, S} where {T, S}
    betas::Vector{<:Real}
    params::MpsParameters
    layers::MpoLayers
    statistics::Dict

    function MpsContractor{T}(peps, betas, params) where T
        new(peps, betas, params, MpoLayers(layout(peps), peps.ncols), Dict())
    end
end
strategy(ctr::MpsContractor{T}) where {T} = T

function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EnergyGauges}
    main = Dict{Site, Sites}(i => (-1//6, 0, 3//6, 4//6) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-3//6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (3//6, 4//6) for i ∈ 1:ncols), right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EnergyGauges}
    MpoLayers(
        Dict(site(i) => (-1//6, 0, 3//6, 4//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (3//6, 4//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end

function MpoLayers(::Type{T}, ncols::Int) where T <: Square{GaugesEnergy}
    main = Dict{Site, Sites}(i => (-4//6, -1//2, 0, 1//6) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-3//6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (1//6,) for i ∈ 1:ncols), right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{GaugesEnergy}
    MpoLayers(
        Dict(site(i) => (-4//6, -1//2, 0, 1//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1//6,) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end

function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EngGaugesEng}
    main = Dict{Site, Sites}(i => (-2//5, -1//5, 0, 1//5, 2//5) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-4//5, -1//5, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (1//5, 2//5) for i ∈ 1:ncols), right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EngGaugesEng}
    MpoLayers(
        Dict(site(i) => (-2//5, -1//5, 0, 1//5, 2//5) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1//5, 2//5) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-4//5, -1//5, 0) for i ∈ 1//2:1//2:ncols)
    )
end

@memoize Dict function mpo(
    ctr::MpsContractor{T}, layers::Dict, r::Int, indβ::Int
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

function dressed_mps(ctr::MpsContractor{T}, i::Int) where T <: AbstractStrategy
    dressed_mps(ctr, i, length(ctr.betas))
end

@memoize Dict function dressed_mps(
    ctr::MpsContractor{T}, i::Int, indβ::Int
) where T <: AbstractStrategy
    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.dress, i, indβ)
    W * ψ
end

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

function conditional_probability(ctr::MpsContractor{S}, w::Vector{Int}) where S
    conditional_probability(layout(ctr.peps), ctr, w)
end

function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, state::Vector{Int},
) where {T <: Square, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j = node_from_index(ctr.peps, length(state)+1)
    ∂v = boundary_state(ctr.peps, state, (i, j))

    L = left_env(ctr, i, ∂v[1:j-1], indβ)
    R = right_env(ctr, i, ∂v[(j+2):(ctr.peps.ncols+1)], indβ)
    M = dressed_mps(ctr, i, indβ)[j]

    L ./= maximum(abs.(L))
    R ./= maximum(abs.(R))
    M ./= maximum(abs.(M))

    @tensor LM[y, z] := L[x] * M[x, y, z]
    eng_local = local_energy(ctr.peps, (i, j))

    pl = projector(ctr.peps, (i, j), (i, j-1))
    eng_pl = interaction_energy(ctr.peps, (i, j), (i, j-1))
    eng_left = @view eng_pl[pl[:], ∂v[j]]

    pu = projector(ctr.peps, (i, j), (i-1, j))
    eng_pu = interaction_energy(ctr.peps, (i, j), (i-1, j))
    eng_up = @view eng_pu[pu[:], ∂v[j+1]]

    en = eng_local .+ eng_left .+ eng_up
    en_min = minimum(en)
    loc_exp = exp.(-β .* (en .- en_min))

    pr = projector(ctr.peps, (i, j), (i, j+1))
    pd = projector(ctr.peps, (i, j), (i+1, j))

    bnd_exp = dropdims(sum(LM[pd[:], :] .* R[:, pr[:]]', dims=2), dims=2)
    probs = loc_exp .* bnd_exp
    push!(ctr.statistics, state => error_measure(probs))
    normalize_probability(probs)
end

# TODO: rewrite this using brodcasting
function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, state::Vector{Int},
) where {T <: SquareStar, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j = node_from_index(ctr.peps, length(state)+1)
    ∂v = boundary_state(ctr.peps, state, (i, j))

    L = left_env(ctr, i, ∂v[1:2*j-2], indβ)
    R = right_env(ctr, i, ∂v[(2*j+3):(2*ctr.peps.ncols+2)], indβ)
    ψ = dressed_mps(ctr, i, indβ)
    MX, M = ψ[j-1//2], ψ[j]
    @tensor LMX[y, z] := L[x] * MX[x, y, z]

    eng_local = local_energy(ctr.peps, (i, j))
    pl = projector(ctr.peps, (i, j), (i, j-1))
    eng_pl = interaction_energy(ctr.peps, (i, j), (i, j-1))
    eng_left = @view eng_pl[pl[:], ∂v[2*j]]

    pu = projector(ctr.peps, (i, j), (i-1, j))
    eng_pu = interaction_energy(ctr.peps, (i, j), (i-1, j))
    eng_up = @view eng_pu[pu[:], ∂v[2*j+2]]

    pd = projector(ctr.peps, (i, j), (i-1, j-1))
    eng_pd = interaction_energy(ctr.peps, (i, j), (i-1, j-1))
    eng_diag = @view eng_pd[pd[:], ∂v[2*j+1]]

    en = eng_local .+ eng_left .+ eng_diag .+ eng_up
    en_min = minimum(en)
    loc_exp = exp.(-β .* (en .- en_min))

    p_lb = projector(ctr.peps, (i, j-1), (i+1, j))
    p_rb = projector(ctr.peps, (i, j), (i+1, j-1))
    pr = projector(ctr.peps, (i, j), ((i+1, j+1), (i, j+1), (i-1, j+1)))
    pd = projector(ctr.peps, (i, j), (i+1, j))

    @cast LMX2[b, c, d] := LMX[(b, c), d] (b ∈ 1:maximum(p_lb), c ∈ 1:maximum(p_rb))

    for σ ∈ 1:length(loc_exp)
        lmx = @view LMX2[p_lb[∂v[2*j-1]], p_rb[σ], :]
        m = @view M[:, pd[σ], :]
        r = @view R[:, pr[σ]]
        loc_exp[σ] *= (lmx' * m * r)[]
    end
    push!(ctr.statistics, state => error_measure(loc_exp))
    normalize_probability(loc_exp)
end

function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, state::Vector{Int},
) where {T <: Pegasus, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j, k = node_from_index(ctr.peps, length(state)+1)
    ∂v = boundary_state(ctr.peps, state, (i, j))

    L = left_env(ctr, i, ∂v[1:j-1], indβ)
    R = right_env(ctr, i, ∂v[(j+2):(ctr.peps.ncols+1)], indβ)
    M = dressed_mps(ctr, i, indβ)[j]

    L ./= maximum(abs.(L))
    R ./= maximum(abs.(R))
    M ./= maximum(abs.(M))

    @tensor LM[y, z] := L[x] * M[x, y, z]

    _, (pl1, pl2) = fuse_projectors([projector(ctr.peps, (i, j-1, 2), (i, j, k)) for k ∈ 1:2])
    _, (pu1, pu2) = fuse_projectors([projector(ctr.peps, (i-1, j, 1), (i, j, k)) for k ∈ 1:2])

    pl = [projector(ctr.peps, (i, j, k), (i, j-1 ,2)) for k ∈ 1:2]
    pu = [projector(ctr.peps, (i, j, k), (i-1, j, 1)) for k ∈ 1:2]

    eu = [interaction_energy(ctr.peps, (i, j, k), (i-1, j, 1)) for k ∈ 1:2]
    el = [interaction_energy(ctr.peps, (i, j, k), (i, j-1, 2)) for k ∈ 1:2]

    eng_left = [el[1][pl[1][:], pl2[∂v[j]]], el[2][pl[2][:], pl1[∂v[j]]]]
    eng_up = [eu[1][pu[1][:], pu1[∂v[j+1]]], eu[2][pu[2][:], pu2[∂v[j+1]]]]

    en21 = interaction_energy(ctr.peps, (i, j, 2), (i, j, 1))
    p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
    p12 = projector(ctr.peps, (i, j, 1), (i, j, 2))

    pr = projector(ctr.peps, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
    pd = projector(ctr.peps, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))

    eng_local = [local_energy(ctr.peps, (i, j, k)) for k ∈ 1:2]

    if k == 1
        en = [eng_local[k] .+ eng_left[k] .+ eng_up[k] for k ∈ 1:2]

        ten = reshape(en[2], (:, 1)) .+ en21
        ten_min = minimum(ten)
        ten2 = exp.(-β .* (ten .- ten_min))
        ten3 = zeros(size(ten2, 1), maximum(pr), size(ten2, 2))
        for i ∈ pr ten3[:, i, :] += ten2 end

        RT = R * dropdims(sum(ten3, dims=1), dims=1)
        bnd_exp = dropdims(sum(LM[pd[:], :] .* RT', dims=2), dims=2)
        en_min = minimum(en[1])
        loc_exp = exp.(-β .* (en[1] .- en_min))
    elseif k == 2
        en = eng_local[2] .+ eng_left[2] .+ eng_up[2] .+ en21[p21[:], p12[∂v[end]]]
        en_min = minimum(en)
        loc_exp = exp.(-β .* (en .- en_min))
        bnd_exp = dropdims(sum(LM[pd[:], :] .* R[:, pr[:]]', dims=2), dims=2)
    else
        throw(ArgumentError("Number $k of sub-clusters is incorrect for this $T."))
    end

    probs = loc_exp .* bnd_exp
    push!(ctr.statistics, state => error_measure(probs))
    normalize_probability(probs)
end

function clear_memoize_cache()
    empty!(memoize_cache(left_env))
    empty!(memoize_cache(right_env))
    empty!(memoize_cache(mpo))
    empty!(memoize_cache(mps))
    empty!(memoize_cache(mps_top))
    empty!(memoize_cache(dressed_mps))
    empty!(memoize_cache(spectrum)) # to be remove
end

function error_measure(probs)
    if maximum(probs) <= 0 return 2.0 end
    if minimum(probs) < 0 return abs(minimum(probs)) / maximum(abs.(probs)) end
    return 0.0
end

function MpoLayers(::Type{T}, ncols::Int) where T <: Pegasus
    main, dress, right = Dict(), Dict(), Dict()
    for i ∈ 1:ncols
        push!(main, i => (-1//3, 0, 1//3))
        push!(dress, i => (1//3))
        push!(right, i => (0, ))
    end
    MpoLayers(main, dress, right)
end

function update_gauges!(ctr::MpsContractor{T}, row::Site, indβ::Int) where T
    clm = ctr.layers.main
    ψ_top = mps_top(ctr, row, indβ)
    ψ_bot = mps(ctr, row + 1, indβ)
    for i ∈ ψ_top.sites
        n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        n_top = PEPSNode(row + clm[i][end], i)
        ρ = overlap_density_matrix(ψ_top, ψ_bot, i)
        _, _, scale = LinearAlgebra.LAPACK.gebal!('S', ρ)
        push!(ctr.peps.gauges.data, n_top => 1 ./ scale, n_bot => scale)
    end

    for ind ∈ 1:indβ
        for i ∈ row:ctr.peps.nrows delete!(memoize_cache(mps_top), (ctr, i, ind)) end
        for i ∈ 1:row+1 delete!(memoize_cache(mps), (ctr, i, ind)) end
        delete!(memoize_cache(mpo), (ctr, ctr.layers.main, row, ind))
        delete!(memoize_cache(mpo), (ctr, ctr.layers.dress, row, ind))
        delete!(memoize_cache(mpo), (ctr, ctr.layers.right, row, ind))
    end
end
