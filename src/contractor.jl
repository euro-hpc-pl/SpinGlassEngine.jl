export
    Basic,
    Annealing,
    MpoLayers,
    MpsParameters,
    MpsContractor

abstract type AbstractContractor end
abstract type AbstractStrategy end

struct Basic <: AbstractStrategy end
struct Annealing <: AbstractStrategy end

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


@memoize function mpo(
    ctr::MpsContractor{T}, 
    layers::Dict, 
    r::Int,
    β::Real
) where T <: AbstractStrategy

    W = Dict()
    # Threads.@threads for (j, coor) ∈ layers
    for (j, coor) ∈ layers
        push!(W, j => Dict(dr => tensor(ctr.peps, PEPSNode(r + dr, j), β) for dr ∈ coor))
    end
    Mpo(W)
end

# IdentityMps to be change or remove
IdentityMps(peps::PEPSNetwork{T, S}) where {T<: Square, S} =
Mps(Dict(j => ones(1, 1, 1) for j ∈ 1:peps.ncols))


function IdentityMps(peps::PEPSNetwork{T, S}) where {T <: SquareStar, S}
    id = Dict()
    for i ∈ 1//2 : 1//2 : peps.ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(id, ii => ones(1, 1, 1))
    end
    Mps(id)
end


@memoize function mps(contractor::MpsContractor{Basic}, i::Int, β::Real) 
    if i > contractor.peps.nrows return IdentityMps(contractor.peps) end  
    ψ = mps(contractor, i+1, β)
    W = mpo(contractor, contractor.layers.main, i, β)

    ψ0 = dot(W, ψ)
    truncate!(ψ0, :left, contractor.params.bond_dimension)
    compress!(ψ0, W, ψ,
            contractor.params.bond_dimension,
            contractor.params.variational_tol,
            contractor.params.max_num_sweeps)
    ψ0
end

### Wybor strategii zwezania boundary mps-a przy pomocy typu MpsContractor
# @memoize function mps(contractor::MpsContractor, i::Int, β::Real) where T <: Number
#     if i > contractor.peps.nrows return IdentityMps(contractor.peps) end  
#     ψ = mps(contractor, i+1, β)
#     W = mpo(contractor, contractor.layers.main, i, β)

#     ψ0 = mps(contractor, i, wieksze beta, lub random/identycznosc/etc..)
#     compress!(ψ0, W, ψ,
#             contractor.params.bond_dimension,
#             contractor.params.variational_tol,
#             contractor.params.max_num_sweeps)
#     ψ0
# end


dressed_mps(contractor::MpsContractor{T}, i::Int) where T <: AbstractStrategy = 
dressed_mps(contractor, i, last(contractor.betas))


@memoize Dict function dressed_mps(
    contractor::MpsContractor{T},
    i::Int, 
    β::Real
) where T <: AbstractStrategy
    ψ = mps(contractor, i+1, β)
    W = mpo(contractor, contractor.layers.dress, i, β)
    W * ψ
end


@memoize Dict function right_env(
    contractor::MpsContractor{T},
    i::Int, 
    ∂v::Vector{Int}, 
    β::Real
) where T <: AbstractStrategy

    l = length(∂v)
    if l == 0 return ones(1, 1) end

    R̃ = right_env(contractor, i, ∂v[2:l], β)
    ϕ = dressed_mps(contractor, i, β)
    W = mpo(contractor, contractor.layers.right, i, β)
    k = length(ϕ.sites)
    site = ϕ.sites[k-l+1]
    M̃ = W[site]
    M = ϕ[site]

    RR = _update_reduced_env_right(R̃, ∂v[1], M̃, M)

    ls_mps = _left_nbrs_site(site, ϕ.sites)
    ls = _left_nbrs_site(site, W.sites)

    while ls > ls_mps
        M0 = W[ls][0]  ## struktura danych w mpo ???
        @tensor RR[x, y] := M0[y, z] * RR[x, z]
        ls = _left_nbrs_site(ls, W.sites)
    end
    RR
end


function _update_reduced_env_right(RE, m::Int, M::Dict, B)
    kk = sort(collect(keys(M)))
    Mt = M[kk[1]]
    K = @view Mt[m, :]

    for ii ∈ kk[2:end]
        if ii == 0 break end
        Mm = M[ii]
        @tensor K[a] := K[b] * Mm[b, a]
    end

    M0 = M[0]  # assume convention that we end at site tensor
    @tensor R[x, y] := K[d] * M0[y, d, β, γ] * 
                       B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
    R
end


@memoize Dict function left_env(
    contractor::MpsContractor{T},
    i::Int, 
    ∂v::Vector{Int}, 
    β::Real
) where T
    l = length(∂v)
    if l == 0 return ones(1) end
    L̃ = left_env(contractor, i, ∂v[1:l-1], β)
    ϕ = dressed_mps(contractor, i, β)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]
    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    L
end


function conditional_probability(contractor::MpsContractor{S}, w::Vector{Int}) where S
    T = layout(contractor.peps)
    β = last(contractor.betas)
    conditional_probability(T, contractor, w, β)
end


function conditional_probability(::Type{T}, 
    contractor::MpsContractor{S}, 
    w::Vector{Int}, 
    β::Real
) where {T <: Square, S}

    network = contractor.peps
    i, j = node_from_index(network, length(w)+1)
    ∂v = boundary_state(network, w, (i, j))

    L = left_env(contractor, i, ∂v[1:j-1], β)
    R = right_env(contractor, i, ∂v[j+2 : network.ncols+1], β)
    ψ = dressed_mps(contractor, i, β)
    M = ψ.tensors[j]

    A = reduced_site_tensor(network, (i, j), ∂v[j], ∂v[j+1], β)

    @tensor prob[σ] := L[x] * M[x, d, y] * A[r, d, σ] *
                       R[y, r] order = (x, d, r, y)

    normalize_probability(prob)
end


function conditional_probability(::Type{T}, 
    contractor::MpsContractor{S}, 
    w::Vector{Int}, 
    β::Real
) where {T <: SquareStar, S}

    network = contractor.peps
    i, j = node_from_index(network, length(w)+1)
    ∂v = boundary_state(network, w, (i, j))

    L = left_env(contractor, i, ∂v[1:2*j-2], β)
    R = right_env(contractor, i, ∂v[2*j+3 : 2*network.ncols+2], β)
    ψ = dressed_mps(contractor, i, β)
    MX, M = ψ[j-1//2], ψ[j]

    A = reduced_site_tensor(network, (i, j), ∂v[2*j-1], ∂v[2*j], ∂v[2*j+1], ∂v[2*j+2], β)

    @tensor prob[σ] := L[x] * MX[x, m, y] * M[y, l, z] * R[z, k] *
                        A[k, l, m, σ] order = (x, y, z, k, l, m)

    normalize_probability(prob)
end


function reduced_site_tensor(
    network::PEPSNetwork{T, S},
    v::Node,
    l::Int,
    u::Int,
    β::Real
) where {T, S}

    i, j = v
    eng_local = local_energy(network, v)
    pl = projector(network, v, (i, j-1))
    # pl 1d -> 2d
    eng_pl = interaction_energy(network, v, (i, j-1))
    @matmul eng_left[x] := sum(y) pl[x, y] * eng_pl[y, $l]
    
    pu = projector(network, v, (i-1, j))
    eng_pu = interaction_energy(network, v, (i-1, j))
    @matmul eng_up[x] := sum(y) pu[x, y] * eng_pu[y, $u]

    en = eng_local .+ eng_left .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    pr = projector(network, v, (i, j+1))
    pd = projector(network, v, (i+1, j))
    @cast A[r, d, σ] := pr[σ, r] * pd[σ, d] * loc_exp[σ]
    A
end


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
    eng_pl = interaction_energy(network, v, (i, j-1))
    @matmul eng_left[x] := sum(y) pl[x, y] * eng_pl[y, $l]

    pd = projector(network, v, (i-1, j-1))
    eng_pd = interaction_energy(network, v, (i-1, j-1))
    @matmul eng_diag[x] := sum(y) pd[x, y] * eng_pd[y, $d]
    
    pu = projector(network, v, (i-1, j))
    eng_pu = interaction_energy(network, v, (i-1, j))
    @matmul eng_up[x] := sum(y) pu[x, y] * eng_pu[y, $u]

    en = eng_local .+ eng_left .+ eng_diag .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    p_lb = projector(network, (i, j-1), (i+1, j))
    p_rb = projector(network, (i, j), (i+1, j-1))
    pr = projector(network, v, ((i+1, j+1), (i, j+1), (i-1, j+1)))
    pd = projector(network, v, (i+1, j))

    @cast A[r, d, (k̃, k), σ] := p_rb[σ, k] * p_lb[$ld, k̃] * pr[σ, r] * 
                                pd[σ, d] * loc_exp[σ]
    A
end
