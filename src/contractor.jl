export SVDTruncate, MPSAnnealing, MpoLayers, MpsParameters, MpsContractor

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

struct MpsContractor{T <: AbstractStrategy} <: AbstractContractor
    peps::PEPSNetwork{T, S} where {T, S}
    betas::Vector{Real}
    params::MpsParameters
    layers::MpoLayers

    function MpsContractor{T}(peps, betas, params) where T
        new(peps, betas, params, MpoLayers(layout(peps), peps.ncols))
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

@memoize function mpo(
    ctr::MpsContractor{T}, layers::Dict, r::Int, indβ::Int
) where T <: AbstractStrategy
    sites = collect(keys(layers))
    tensors = Vector{Dict}(undef, length(sites))

    #Threads.@threads for i ∈ 1:length(sites)
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
IdentityQMps(peps::PEPSNetwork{T, S}) where {T <: Square, S} =
QMps(Dict(j => ones(1, 1, 1) for j ∈ 1:peps.ncols))

function IdentityQMps(peps::PEPSNetwork{T, S}, Dmax::Int, loc_dim) where {T, S}
    id = Dict{Int, Array{Float64, 3}}()
    for i ∈ 2 : peps.ncols-1 push!(id, i => zeros(Dmax, loc_dim[i], Dmax)) end

    push!(id, 1 => zeros(1, loc_dim[1], Dmax))
    push!(id, peps.ncols => zeros(Dmax, loc_dim[peps.ncols], 1))

    for i ∈ 2 : peps.ncols-1 id[i][1, :, 1] .= 1 / sqrt(loc_dim[i]) end
    QMps(id)
end

function IdentityQMps(peps::PEPSNetwork{T, S}) where {T <: SquareStar, S}
    id = Dict()
    for i ∈ 1//2 : 1//2 : peps.ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(id, ii => ones(1, 1, 1))
    end
    QMps(id)
end

@memoize function mps(contractor::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    if i > contractor.peps.nrows return IdentityQMps(contractor.peps) end

    ψ = mps(contractor, i+1, indβ)
    W = mpo(contractor, contractor.layers.main, i, indβ)

    ψ0 = dot(W, ψ)
    truncate!(ψ0, :left, contractor.params.bond_dimension)
    compress!(
        ψ0, W, ψ, contractor.params.bond_dimension, contractor.params.variational_tol,
        contractor.params.max_num_sweeps
    )
    ψ0
end

@memoize function mps(contractor::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
    if i > contractor.peps.nrows return IdentityQMps(contractor.peps) end

    ψ = mps(contractor, i+1, indβ)
    W = mpo(contractor, contractor.layers.main, i, indβ)

    if indβ > 1
        ψ0 = mps(contractor, i, indβ-1)
    else
        bd = contractor.params.bond_dimension
        ψ0 = IdentityQMps(contractor.peps, bd, local_dims(W, :up))
        canonise!(ψ0, :left)
    end
    compress!(
            ψ0, W, ψ, contractor.params.bond_dimension, contractor.params.variational_tol,
            contractor.params.max_num_sweeps
    )
    ψ0
end
dressed_mps(contractor::MpsContractor{T}, i::Int) where T <: AbstractStrategy =
dressed_mps(contractor, i, length(contractor.betas))

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

# function _update_reduced_env_right(
#     K::AbstractArray{Float64, 1},
#     RE::AbstractArray{Float64, 2},
#     M::SparseSiteTensor,
#     B::AbstractArray{Float64, 3}
# )
#     R = zeros(size(B, 1), maximum(M.projs[1]))
#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         re = @view RE[:, M.projs[3][σ]]
#         b = @view B[:, M.projs[4][σ], :]
#         R[:, M.projs[1][σ]] += (lexp * K[M.projs[2][σ]]) .* (b * re)
#     end
#     R
# end

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
    # to be written
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
    normalize_probability(loc_exp .* bnd_exp)
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
    normalize_probability(loc_exp)
end
