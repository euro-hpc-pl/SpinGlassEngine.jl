export
    SVDTruncate,
    MPSAnnealing,
    MpoLayers,
    MpsParameters,
    MpsContractor

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

    MpsContractor{T}(peps, betas, params) where T =
        new(peps, betas, params, MpoLayers(layout(peps), peps.ncols))
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
    ctr::MpsContractor{T}, 
    layers::Dict, 
    r::Int,
    indβ::Int
) where T <: AbstractStrategy

    sites = collect(keys(layers))
    tensors = Vector{Dict}(undef, length(sites))

    #Threads.@threads for i ∈ 1:length(sites)
    for i ∈ 1:length(sites)
        j = sites[i]
        coor = layers[j]
        tensors[i] = Dict(dr => tensor(ctr.peps, PEPSNode(r + dr, j), ctr.betas[indβ])
                            for dr ∈ coor)
    end

    Mpo(Dict(sites .=> tensors))
end                        

# IdentityMps to be change or remove
IdentityMps(peps::PEPSNetwork{T, S}) where {T <: Square, S} =
Mps(Dict(j => ones(1, 1, 1) for j ∈ 1:peps.ncols))


function IdentityMps(peps::PEPSNetwork{T, S}, Dmax::Int, loc_dim::Vector{Int}) where {T <: Square, S}
    id = Dict()
    for i ∈ 2 : peps.ncols-1
        push!(id, i => zeros(Dmax, loc_dim[i], Dmax))
    end
    push!(id, 1 => zeros(1, loc_dim[1], Dmax))
    push!(id, peps.ncols => zeros(Dmax, loc_dim[peps.ncols], 1))
    for i ∈ 2 : peps.ncols-1
        id[i][1, :, 1] = 1 ./sqrt(loc_dim[i])
    end
    Mps(id)
end



function IdentityMps(peps::PEPSNetwork{T, S}) where {T <: SquareStar, S}
    id = Dict()
    for i ∈ 1//2 : 1//2 : peps.ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(id, ii => ones(1, 1, 1))
    end
    Mps(id)
end


@memoize function mps(contractor::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    if i > contractor.peps.nrows return IdentityMps(contractor.peps) end

    ψ = mps(contractor, i+1, indβ)
    W = mpo(contractor, contractor.layers.main, i, indβ)

    ψ0 = dot(W, ψ)
    truncate!(ψ0, :left, contractor.params.bond_dimension)
    compress!(
            ψ0,
            W,
            ψ,
            contractor.params.bond_dimension,
            contractor.params.variational_tol,
            contractor.params.max_num_sweeps)
    ψ0
end


@memoize function mps(contractor::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
    if i > contractor.peps.nrows return IdentityMps(contractor.peps) end

    ψ = mps(contractor, i+1, indβ)
    W = mpo(contractor, contractor.layers.main, i, indβ)

    if indβ > 1
        ψ0 = mps(contractor, i, indβ-1)
    else
        ld = local_dims(W, :up)
        ψ0 = IdentityMps(contractor.peps, contractor.params.bond_dimension, ld)
        canonize!(ψ0)
    end
    compress!(
            ψ0,
            W,
            ψ,
            contractor.params.bond_dimension,
            contractor.params.variational_tol,
            contractor.params.max_num_sweeps)
    ψ0
end


dressed_mps(contractor::MpsContractor{T}, i::Int) where T <: AbstractStrategy = 
dressed_mps(contractor, i, length(contractor.betas))


@memoize Dict function dressed_mps(
    contractor::MpsContractor{T},
    i::Int, 
    indβ::Int
) where T <: AbstractStrategy
    ψ = mps(contractor, i+1, indβ)
    W = mpo(contractor, contractor.layers.dress, i, indβ)
    W * ψ
end


@memoize Dict function right_env(
    contractor::MpsContractor{T},
    i::Int,
    ∂v::Vector{Int},
    indβ::Int
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
        M0 = W[ls][0]  # make this consistent
        @tensor RR[x, y] := M0[y, z] * RR[x, z]
        ls = _left_nbrs_site(ls, W.sites)
    end
    RR
end


function _update_reduced_env_right(
    RE::AbstractArray{Float64, 2}, 
    m::Int, 
    M::Dict, 
    B::AbstractArray{Float64, 3}
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
    K::AbstractArray{Float64, 1},  # D 
    RE::AbstractArray{Float64, 2}, # chi x D
    M::AbstractArray{Float64, 4},  # D x D x D x D
    B::AbstractArray{Float64, 3}  # chi x D x chi
)
    @tensor R[x, y] := K[d] * M[y, d, β, γ] *
                       B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
    R  # O(D^4 + D^3 chi + D^2 chi^2);  pamiec D^4 + D chi^2 + D^2 chi
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
#     R  # O(N chi^2 D); pamiec chi^2 D, nie ma D^2
#     #  dla chimera N = D^2; O(D^3 chi^2)
# end


function _update_reduced_env_right(
    K::AbstractArray{Float64, 1},
    RE::AbstractArray{Float64, 2},
    M::SparseSiteTensor,
    B::AbstractArray{Float64, 3}
)
    @tensor REB[x, y, β] := B[x, y, α] * RE[α, β]

    Kloc_exp = M.loc_exp .* vec(K[M.projs[2]])
    s3 = maximum(M.projs[4])
    ind43 = vec(M.projs[4]) .+ ((vec(M.projs[3]) .- 1) .* s3)
    @cast REB2[x, (y, z)] := REB[x, y, z]
    Rσ = REB2[:, ind43]
    R = zeros(size(B, 1), maximum(M.projs[1]))
    for (σ, kl) ∈ enumerate(Kloc_exp)
        R[:, M.projs[1][σ]] += kl .* Rσ[:, σ]
    end
    # R = zeros(size(B, 1), maximum(M.projs[1]))
    # for (σ, lexp) ∈ enumerate(M.loc_exp)
    #     R[:, M.projs[1][σ]] += (lexp * K[M.projs[2][σ]]) .* REB[:, M.projs[4][σ], M.projs[3][σ]]
    # end
    R  # O(chi^2 D^2 + N chi); pamiec co najmniej chi D^2 + chi^2 D
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
    contractor::MpsContractor{T},
    i::Int, 
    ∂v::Vector{Int}, 
    indβ::Int
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
    T = layout(contractor.peps)
    conditional_probability(T, contractor, w)
end


function conditional_probability(::Type{T}, 
    contractor::MpsContractor{S}, 
    state::Vector{Int},
) where {T <: Square, S}

    indβ, β = length(contractor.betas), last(contractor.betas)
    i, j = node_from_index(contractor.peps, length(state)+1)
    ∂v = boundary_state(contractor.peps, state, (i, j))

    L = left_env(contractor, i, ∂v[1:j-1], indβ)
    R = right_env(contractor, i, ∂v[j+2 : contractor.peps.ncols+1], indβ)
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

    # Threads.@threads for σ ∈ 1:length(loc_exp)
    # for σ ∈ 1:length(loc_exp)
    #     loc_exp[σ] *= (LM[pd[σ], :]' * R[:, pr[σ]])
    # end
    # normalize_probability(loc_exp)

    # variant replacing for-loop above
    bnd_exp = vec(sum(LM[pd[:], :] .* R[:, pr[:]]', dims=2))
    normalize_probability(loc_exp .* bnd_exp)
end


# to be improved
function conditional_probability(::Type{T},
    contractor::MpsContractor{S}, 
    w::Vector{Int},
) where {T <: SquareStar, S}

    indβ = length(contractor.betas)
    network = contractor.peps
    i, j = node_from_index(network, length(w)+1)
    ∂v = boundary_state(network, w, (i, j))

    L = left_env(contractor, i, ∂v[1:2*j-2], indβ)
    R = right_env(contractor, i, ∂v[2*j+3 : 2*network.ncols+2], indβ)
    ψ = dressed_mps(contractor, i, indβ)
    MX, M = ψ[j-1//2], ψ[j]


    β = contractor.betas[indβ]
    A = reduced_site_tensor(network, (i, j), ∂v[2*j-1], ∂v[2*j], ∂v[2*j+1], ∂v[2*j+2], β)

    @tensor prob[σ] := L[x] * MX[x, m, y] * M[y, l, z] * R[z, k] *
                        A[k, l, m, σ] order = (x, y, z, k, l, m)

    normalize_probability(prob)
end


# to be removed
function reduced_site_tensor(
    network::PEPSNetwork{T, S},
    v::Node,
    ld::Int,
    l::Int,
    d::Int,
    u::Int,
    β::Real
) where {T <: SquareStar, S}

    i, j = v
    eng_local = local_energy(network, v)

    pl = projector(network, v, (i, j-1))
    pl = decode_projector!(pl)
    eng_pl = interaction_energy(network, v, (i, j-1))
    @matmul eng_left[x] := sum(y) pl[x, y] * eng_pl[y, $l]

    pd = projector(network, v, (i-1, j-1))
    pd = decode_projector!(pd)
    eng_pd = interaction_energy(network, v, (i-1, j-1))
    @matmul eng_diag[x] := sum(y) pd[x, y] * eng_pd[y, $d]
    
    pu = projector(network, v, (i-1, j))
    pu = decode_projector!(pu)
    eng_pu = interaction_energy(network, v, (i-1, j))
    @matmul eng_up[x] := sum(y) pu[x, y] * eng_pu[y, $u]

    en = eng_local .+ eng_left .+ eng_diag .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    p_lb = decode_projector!(projector(network, (i, j-1), (i+1, j)))
    p_rb = decode_projector!(projector(network, (i, j), (i+1, j-1)))
    pr = decode_projector!(projector(network, v, ((i+1, j+1), (i, j+1), (i-1, j+1))))
    pd = decode_projector!(projector(network, v, (i+1, j)))

    @cast A[r, d, (k̃, k), σ] := p_rb[σ, k] * p_lb[$ld, k̃] * pr[σ, r] * 
                                pd[σ, d] * loc_exp[σ]
    A
end
