export
    reduced_site_tensor,
    tensor_size,
    tensor,
    right_env,
    left_env,
    dressed_mps,
    mpo, mps 


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::R
) where {S, T, R}
    if v ∈ keys(network.tensors_map)
        tensor(network, v, Val(network.tensors_map[v]))
    else
        ones(1, 1)
    end
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T}, 
    v::R
) where {S, T, R}
    if v ∈ keys(network.tensors_map)
        tensor_size(network, v, Val(network.tensors_map[v]))
    else
        (1, 1)
    end
end


function tensor(
    network::AbstractGibbsNetwork{S, T}, 
    v::S,
    ::Val{:site}
) where {S, T}
    loc_exp = exp.(-network.β .* local_energy(network, v))
    projs = projectors(network, v)
    # tu ma byc decode projector
    @cast A[σ, _] := loc_exp[σ]
    for pv ∈ projs @cast A[σ, (c, γ)] |= A[σ, c] * pv[σ, γ] end 
    B = dropdims(sum(A, dims=1), dims=1)
    reshape(B, size.(projs, 2))
end
 

function tensor_size(
    network::AbstractGibbsNetwork{S, T}, 
    v::S,
    ::Val{:site}
) where {S, T}
    dims = size.(projectors(network, v))
     # tu ma byc decode projector -> max(projecr) da ilosc elementow
    pdims = first.(dims)
    @assert all(σ -> σ == first(pdims), first.(dims))
    last.(dims)
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Rational{Int}, Int},
    ::Val{:central_v}
) where {S, T}
    r, j = v
    i = floor(Int, r)
    connecting_tensor(network, (i, j), (i+1, j))
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Rational{Int}, Int},
    ::Val{:central_v}
) where {S, T}
    r, j = v
    i = floor(Int, r)
    size(interaction_energy(network, (i, j), (i+1, j)))
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    w::Tuple{Int, Rational{Int}},
    ::Val{:central_h}
) where {S, T}
    i, r = w
    j = floor(Int, r)
    connecting_tensor(network, (i, j), (i, j+1))
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T},
    w::Tuple{Int, Rational{Int}},
    ::Val{:central_h}
) where {S, T}
    i, r = w
    j = floor(Int, r)
    size(interaction_energy(network, (i, j), (i, j+1)))
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Rational{Int}, Rational{Int}},
    ::Val{:central_d}
) where {S, T}
    r, s = v
    i = floor(Int, r)
    j = floor(Int, s)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1))
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j))
    @cast A[(u, ũ), (d, d̃)] := NW[u, d] * NE[ũ, d̃] 
    A
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Rational{Int}, Rational{Int}},
    ::Val{:central_d}
) where {S, T}
    r, s = v
    i = floor(Int, r)
    j = floor(Int, s)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    u * ũ, d * d̃
end


function _all_fused_projectors(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Int, Rational{Int}},
) where {S, T}
    i, s = v
    j = floor(Int, s)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    p_lb, p_l, p_lt, p_rb, p_r, p_rt
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Int, Rational{Int}},
    ::Val{:virtual}
) where {S, T}
    p_lb, p_l, p_lt, 
    p_rb, p_r, p_rt = _all_fused_projectors(network, v)

    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v))

    @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]    
    @cast A[l, (ũ, u), r, (d̃, d)] |= B[l, r] * p_lt[l, u] * p_rb[r, d] * 
                                     p_rt[r, ũ] * p_lb[l, d̃]
    A
end 


function tensor_size(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Int, Rational{Int}},
    ::Val{:virtual}
) where {S, T}
    p_lb, p_l, p_lt, 
    p_rb, p_r, p_rt = _all_fused_projectors(network, v)
    (size(p_l, 1), size(p_lt, 2) * size(p_rt, 2),
     size(p_r, 1), size(p_rb, 2) * size(p_lb, 2))
end


function tensor(
    network::AbstractGibbsNetwork{S, T}, 
    v::R,
    ::Val{:gauge_h}
) where {S, T, R}
    Diagonal(network.gauges_data[v])
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T}, 
    v::R,
    ::Val{:gauge_h}
) where {S, T, R}
    u = size(network.gauges_data[v], 1)
    u, u
end


function connecting_tensor(
    network::AbstractGibbsNetwork{S, T},
    v::S,
    w::S
) where {S, T}
    en = interaction_energy(network, v, w)
    exp.(-network.β .* (en .- minimum(en)))
end 


function reduced_site_tensor(
    network::PEPSNetwork,
    v::Tuple{Int, Int},
    l::Int,
    u::Int
)
    i, j = v
    eng_local = local_energy(network, v)
    pl = projector(network, v, (i, j-1))
    eng_pl = interaction_energy(network, v, (i, j-1))
    @matmul eng_left[x] := sum(y) pl[x, y] * eng_pl[y, $l]
    
    pu = projector(network, v, (i-1, j))
    eng_pu = interaction_energy(network, v, (i-1, j))
    @matmul eng_up[x] := sum(y) pu[x, y] * eng_pu[y, $u]

    en = eng_local .+ eng_left .+ eng_up
    loc_exp = exp.(-network.β .* (en .- minimum(en)))

    pr = projector(network, v, (i, j+1))
    pd = projector(network, v, (i+1, j))
    @cast A[r, d, σ] := pr[σ, r] * pd[σ, d] * loc_exp[σ]
    A
end


function reduced_site_tensor(
    network::PEPSNetwork{T},
    v::Tuple{Int, Int},
    ld::Int,
    l::Int,
    d::Int,
    u::Int
) where T <: SquareStar

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
    loc_exp = exp.(-network.β .* (en .- minimum(en)))

    p_lb = projector(network, (i, j-1), (i+1, j))
    p_rb = projector(network, (i, j), (i+1, j-1))
    pr = projector(network, v, ((i+1, j+1), (i, j+1), (i-1, j+1)))
    pd = projector(network, v, (i+1, j))

    @cast A[r, d, (k̃, k), σ] := p_rb[σ, k] * p_lb[$ld, k̃] * pr[σ, r] * 
                                pd[σ, d] * loc_exp[σ]
    A
end


function tensor_size(
    network::PEPSNetwork{T},
    v::Tuple{Int, Int},
    ::Val{:reduced}
) where T <: Square

    i, j = v
    pr = projector(network, v, (i, j+1))
    pd = projector(network, v, (i+1, j))
    @assert size(pr, 1) == size(pr, 1)
    size(pr, 2), size(pd, 2), size(pd, 1)
end 


@memoize function mpo(peps::AbstractGibbsNetwork, layers, r::Int) where {T <: Number}
    W = Dict()
    # Threads.@threads for (j, coor) ∈ layers
    for (j, coor) ∈ layers
        push!(W,
            j => Dict(dr => tensor(peps, (r + dr, j)) for dr ∈ coor)
        )
    end
    Mpo(W)
end


IdentityMps(peps::PEPSNetwork{T}) where T <: Square = Mps(   ## change for pegazus
    Dict(j => ones(1, 1, 1) for j ∈ 1:peps.ncols)
)

function IdentityMps(peps::PEPSNetwork{T}) where T <: SquareStar
    id = Dict()
    for i ∈ 1//2 : 1//2 : peps.ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(id, ii => ones(1, 1, 1))
    end
    Mps(id)
end

@memoize function mps(peps::AbstractGibbsNetwork, i::Int) where {T <: Number}
    if i > peps.nrows return IdentityMps(peps) end  
    ψ = mps(peps, i+1)
    W = mpo(peps, peps.mpo_main, i)
    ψ0 = dot(W, ψ)   # dla rzadkosci nie mozemy tworzyc dot(W, ψ)
    # jako initial guess mozemy probowac wykorzystac mpsy z innych temperatur
    truncate!(ψ0, :left, peps.bond_dim)
    compress!(ψ0, W, ψ, peps.bond_dim, peps.var_tol, peps.sweeps) 
    ψ0
end


@memoize Dict function dressed_mps(peps::AbstractGibbsNetwork, i::Int)
    ψ = mps(peps, i+1)
    W = mpo(peps, peps.mpo_dress, i)
    W * ψ
end


# move contractio to SGTensor ???
@memoize Dict function right_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int}) 
    l = length(∂v)
    if l == 0 return ones(1, 1) end

    R̃ = right_env(peps, i, ∂v[2:l])
    ϕ = dressed_mps(peps, i)
    W = mpo(peps, peps.mpo_right, i)
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
    M0 = M[0]
    Mt = M[-1//2]
    K = @view Mt[m, :]
    @tensor R[x, y] := K[d] * M0[y, d, β, γ] * 
                       B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
    R
end


@memoize Dict function left_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int})
    l = length(∂v)
    if l == 0 return ones(1) end
    L̃ = left_env(peps, i, ∂v[1:l-1])
    ϕ = dressed_mps(peps, i)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]
    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    L
end

