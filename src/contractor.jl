export SVDTruncate, MPSAnnealing, MpoLayers, MpsParameters, MpsContractor, clear_cache,  mps_top, mps

abstract type AbstractContractor end
abstract type AbstractStrategy end

struct SVDTruncate <: AbstractStrategy end
struct MPSAnnealing <: AbstractStrategy end

struct MpoLayers
    main::Dict
    dress::Dict
    right::Dict
end

struct MpsParameters
    bond_dimension::Int
    variational_tol::Real
    max_num_sweeps::Int

    MpsParameters(bd=typemax(Int), ϵ=1E-8, sw=4) = new(bd, ϵ, sw)
end
layout(network::PEPSNetwork{T, S}) where {T, S} = T
sparsity(network::PEPSNetwork{T, S}) where {T, S} = S

mutable struct MpsContractor{T <: AbstractStrategy} <: AbstractContractor
    peps::PEPSNetwork{T, S} where {T, S}
    betas::Vector{Real}
    params::MpsParameters
    layers::MpoLayers
    statistics::Dict

    function MpsContractor{T}(peps, betas, params) where T
        new(peps, betas, params, MpoLayers(layout(peps), peps.ncols), Dict())
    end
end
strategy(ctr::MpsContractor{T}) where {T} = T

function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EnergyGauges}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1:ncols push!(main, i => (-1//6, 0, 3//6, 4//6)) end
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

    for i ∈ 1:ncols push!(right, i => (-3//6, 0)) end
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, dress, right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EnergyGauges}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1//2 : 1//2 : ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(main, ii => (-1//6, 0, 3//6, 4//6))
        push!(dress, ii => (3//6, 4//6))
        push!(right, ii => (-3//6, 0))
    end
    MpoLayers(main, dress, right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: Square{GaugesEnergy}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1:ncols push!(main, i => (-4//6, -1//2, 0, 1//6)) end
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    dress = Dict(i => (1//6,) for i ∈ 1:ncols)

    for i ∈ 1:ncols push!(right, i => (-3//6, 0)) end
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, dress, right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{GaugesEnergy}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1//2 : 1//2 : ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(main, ii => (-4//6, -1//2, 0, 1//6))
        push!(dress, ii => (1//6))
        push!(right, ii => (-3//6, 0))
    end
    MpoLayers(main, dress, right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EngGaugesEng}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1:ncols push!(main, i => (-2//5, -1//5, 0, 1//5, 2//5)) end
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    dress = Dict(i => (1//5, 2//5) for i ∈ 1:ncols)

    for i ∈ 1:ncols push!(right, i => (-4//5, -1//5, 0)) end
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, dress, right)
end

function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EngGaugesEng}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1//2 : 1//2 : ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(main, ii => (-2//5, -1//5, 0, 1//5, 2//5))
        push!(dress, ii => (1//5, 2//5))
        push!(right, ii => (-4//5, -1//5, 0))
    end
    MpoLayers(main, dress, right)
end

@memoize function mpo(
    ctr::MpsContractor{T}, layers::Dict, r::Int, indβ::Int
) where T <: AbstractStrategy
    sites = collect(keys(layers))
    tensors = Vector{Dict}(undef, length(sites))

    #Threads.@threads for i ∈ 1:length(sites) #TODO: does this make sense here?
    for i ∈ 1:length(sites)
        j = sites[i]
        coor = layers[j]
        tensors[i] = Dict(
                        dr => tensor(ctr.peps, PEPSNode(r + dr, j), ctr.betas[indβ])
                        for dr ∈ coor
                    )
    end
    QMpo(Dict(sites .=> tensors))
end


@memoize function mps_top(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
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


@memoize function mps(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
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

@memoize function mps(ctr::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
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

function dressed_mps(contractor::MpsContractor{T}, i::Int) where T <: AbstractStrategy
    dressed_mps(contractor, i, length(contractor.betas))
end

@memoize Dict function dressed_mps(
    contractor::MpsContractor{T}, i::Int, indβ::Int
) where T <: AbstractStrategy
    ψ = mps(contractor, i+1, indβ)
    W = mpo(contractor, contractor.layers.dress, i, indβ)
    W * ψ
end

@memoize Dict function right_env(
    contractor::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T <: AbstractStrategy
    l = length(∂v)
    if l == 0 return ones(1, 1) end

    R̃ = right_env(contractor, i, ∂v[2:l], indβ)
    ϕ = dressed_mps(contractor, i, indβ)
    W = mpo(contractor, contractor.layers.right, i, indβ)
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
    Mt = M[kk[1]]
    K = @view Mt[m, :]

    for ii ∈ kk[2:end]
        if ii == 0 break end
        Mm = M[ii]
        @tensor K[a] := K[b] * Mm[b, a]
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
    contractor::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T
    l = length(∂v)
    if l == 0 return ones(1) end
    L̃ = left_env(contractor, i, ∂v[1:l-1], indβ)
    ϕ = dressed_mps(contractor, i, indβ)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]
    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    L
end

function conditional_probability(contractor::MpsContractor{S}, w::Vector{Int}) where S
    conditional_probability(layout(contractor.peps), contractor, w)
end

function conditional_probability(
    ::Type{T}, contractor::MpsContractor{S}, state::Vector{Int},
) where {T <: Square, S}
    indβ, β = length(contractor.betas), last(contractor.betas)
    i, j = node_from_index(contractor.peps, length(state)+1)
    ∂v = boundary_state(contractor.peps, state, (i, j))

    L = left_env(contractor, i, ∂v[1:j-1], indβ)
    R = right_env(contractor, i, ∂v[(j+2):(contractor.peps.ncols+1)], indβ)
    M = dressed_mps(contractor, i, indβ)[j]

    L = L ./ maximum(abs.(L))
    R = R ./ maximum(abs.(R))
    M = M ./ maximum(abs.(M))

    @tensor LM[y, z] := L[x] * M[x, y, z]

    eng_local = local_energy(contractor.peps, (i, j))

    pl = projector(contractor.peps, (i, j), (i, j-1))
    eng_pl = interaction_energy(contractor.peps, (i, j), (i, j-1))
    eng_left = @view eng_pl[pl[:], ∂v[j]]

    pu = projector(contractor.peps, (i, j), (i-1, j))
    eng_pu = interaction_energy(contractor.peps, (i, j), (i-1, j))
    eng_up = @view eng_pu[pu[:], ∂v[j+1]]

    en = eng_local .+ eng_left .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    pr = projector(contractor.peps, (i, j), (i, j+1))
    pd = projector(contractor.peps, (i, j), (i+1, j))

    bnd_exp = dropdims(sum(LM[pd[:], :] .* R[:, pr[:]]', dims=2), dims=2)
    probs = loc_exp .* bnd_exp
    # println(minimum(probs), maximum(probs))
    push!(contractor.statistics, state => error_measure(probs))
    normalize_probability(probs)
end

# TODO: rewrite this using brodcasting
function conditional_probability(
    ::Type{T}, contractor::MpsContractor{S}, state::Vector{Int},
) where {T <: SquareStar, S}
    indβ, β = length(contractor.betas), last(contractor.betas)
    i, j = node_from_index(contractor.peps, length(state)+1)
    ∂v = boundary_state(contractor.peps, state, (i, j))

    L = left_env(contractor, i, ∂v[1:2*j-2], indβ)
    R = right_env(contractor, i, ∂v[(2*j+3):(2*contractor.peps.ncols+2)], indβ)
    ψ = dressed_mps(contractor, i, indβ)
    MX, M = ψ[j-1//2], ψ[j]
    @tensor LMX[y, z] := L[x] * MX[x, y, z]

    eng_local = local_energy(contractor.peps, (i, j))
    pl = projector(contractor.peps, (i, j), (i, j-1))
    eng_pl = interaction_energy(contractor.peps, (i, j), (i, j-1))
    eng_left = @view eng_pl[pl[:], ∂v[2*j]]

    pu = projector(contractor.peps, (i, j), (i-1, j))
    eng_pu = interaction_energy(contractor.peps, (i, j), (i-1, j))
    eng_up = @view eng_pu[pu[:], ∂v[2*j+2]]

    pd = projector(contractor.peps, (i, j), (i-1, j-1))
    eng_pd = interaction_energy(contractor.peps, (i, j), (i-1, j-1))
    eng_diag = @view eng_pd[pd[:], ∂v[2*j+1]]

    en = eng_local .+ eng_left .+ eng_diag .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    p_lb = projector(contractor.peps, (i, j-1), (i+1, j))
    p_rb = projector(contractor.peps, (i, j), (i+1, j-1))
    pr = projector(contractor.peps, (i, j), ((i+1, j+1), (i, j+1), (i-1, j+1)))
    pd = projector(contractor.peps, (i, j), (i+1, j))

    @cast LMX2[b, c, d] := LMX[(b, c), d] (b ∈ 1:maximum(p_lb), c ∈ 1:maximum(p_rb))

    for σ ∈ 1:length(loc_exp)
        lmx = @view LMX2[p_lb[∂v[2*j-1]], p_rb[σ], :]
        m = @view M[:, pd[σ], :]
        r = @view R[:, pr[σ]]
        loc_exp[σ] *= (lmx' * m * r)[]
    end
    push!(contractor.statistics, state => error_measure(loc_exp))
    normalize_probability(loc_exp)
end


function conditional_probability(
    ::Type{T}, contractor::MpsContractor{S}, state::Vector{Int},
) where {T <: Pegasus, S}
    indβ, β = length(contractor.betas), last(contractor.betas)
    i, j, k = node_from_index(contractor.peps, length(state)+1)
    println(i, j, k)
    ∂v = boundary_state(contractor.peps, state, (i, j))
    println(∂v)

    L = left_env(contractor, i, ∂v[1:j-1], indβ)
    R = right_env(contractor, i, ∂v[(j+2):(contractor.peps.ncols+1)], indβ)
    M = dressed_mps(contractor, i, indβ)[j]

    L = L ./ maximum(abs.(L))
    R = R ./ maximum(abs.(R))
    M = M ./ maximum(abs.(M))

    @tensor LM[y, z] := L[x] * M[x, y, z]

    eng_local = local_energy(contractor.peps, (i, j))

    pl = projector(contractor.peps, (i, j), (i, j-1))
    eng_pl = interaction_energy(contractor.peps, (i, j), (i, j-1))
    eng_left = @view eng_pl[pl[:], ∂v[j]]

    pu = projector(contractor.peps, (i, j), (i-1, j))
    eng_pu = interaction_energy(contractor.peps, (i, j), (i-1, j))
    eng_up = @view eng_pu[pu[:], ∂v[j+1]]

    en = eng_local .+ eng_left .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    pr = projector(contractor.peps, (i, j), (i, j+1))
    pd = projector(contractor.peps, (i, j), (i+1, j))

    bnd_exp = dropdims(sum(LM[pd[:], :] .* R[:, pr[:]]', dims=2), dims=2)
    probs = loc_exp .* bnd_exp
    # println(minimum(probs), maximum(probs))
    push!(contractor.statistics, state => error_measure(probs))
    normalize_probability(probs)
end


function clear_cache()
    empty!(memoize_cache(left_env))
    empty!(memoize_cache(right_env))
    empty!(memoize_cache(mpo))
    empty!(memoize_cache(mps))
    empty!(memoize_cache(dressed_mps))
end


function error_measure(probs)
    if maximum(probs) <= 0
        return 2.
    end
    if minimum(probs) < 0
        return abs(minimum(probs)) / maximum(abs.(probs))
    end
    return 0.
end


function MpoLayers(::Type{T}, ncols::Int) where T <: Pegasus
    main, dress, right = Dict(), Dict(), Dict()
    for i ∈ 1 : ncols
        push!(main, i => (-1//3, 0, 1//3))
        push!(dress, i => (1//3))
        push!(right, i => (0, ))
    end
    MpoLayers(main, dress, right)
end

