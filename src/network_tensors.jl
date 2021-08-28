export
    tensor_assignment,
    tensor_species_map!,
    reduced_site_tensor,
    tensor_size,
    tensor,
    right_env,
    left_env,
    dressed_mps,
    mpo, mps


tensor_assignment(
    network::AbstractGibbsNetwork{S, T},
    s::Symbol
) where {S, T} = tensor_assignment(network, Val(s))


tensor_assignment(
    network::AbstractGibbsNetwork{S, T},
    ::Val{:site} 
) where {S, T} = Dict(
    (i, j) => :site for i ∈ 1:network.nrows, j ∈ 1:network.ncols
)


tensor_assignment(
    network::AbstractGibbsNetwork{S, T},
    ::Val{:central_h} 
) where {S, T} = Dict(
    (i, j + 1//2) => :central_h for i ∈ 1:network.nrows, j ∈ 1:network.ncols
)

 
tensor_assignment(
    network::AbstractGibbsNetwork{S, T},
    ::Val{:central_v} 
) where {S, T} = Dict(
    (i + 1//2, j) => :central_v for i ∈ 1:network.nrows-1, j ∈ 1:network.ncols
)


tensor_assignment(
    network::PEPSNetwork,
    ::Val{:gauge_h} 
) = Dict((i + δ, j) => :gauge_h 
    for i ∈ 1:network.nrows-1, j ∈ 1:network.ncols, δ ∈ (1//6, 2//6, 4//6, 5//6)
)


tensor_assignment(
    network::FusedNetwork,
    ::Val{:gauge_h} 
) = Dict((i + δ, r) => :gauge_h 
    for i ∈ 1:network.nrows-1, r ∈ 1:1//2:network.ncols, δ ∈ (1//6, 2//6, 4//6, 5//6)
)


tensor_assignment(
    network::FusedNetwork,
    ::Val{:virtual} 
) = Dict(
    (i, j + 1//2) => :virtual for i ∈ 1:network.nrows, j ∈ 1:network.ncols-1
)


tensor_assignment(
    network::FusedNetwork,
    ::Val{:central_d} 
) = Dict(
    (i + 1//2, j + 1//2) => :central_d for i ∈ 1:network.nrows-1, j ∈ 1:network.ncols-1
)



function tensor_species_map!(
    network::AbstractGibbsNetwork{S, T}, 
    tensor_types::NTuple{N, Symbol}
) where {S, T, N}
    for type ∈ tensor_types
        push!(network.tensor_spiecies, tensor_assignment(network, type)...) 
    end 
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::R
) where {S, T, R}
    if v ∈ keys(network.tensor_spiecies)
        tensor(network, v, Val(network.tensor_spiecies[v]))
    else
        ones(1, 1, 1, 1)
    end
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T}, 
    v::R
) where {S, T, R}
    if v ∈ keys(network.tensor_spiecies)
        tensor_size(network, v, Val(network.tensor_spiecies[v]))
    else
        (1, 1, 1, 1)
    end
end


function tensor(
    network::AbstractGibbsNetwork{S, T}, 
    v::S,
    ::Val{:site}
) where {S, T}
    loc_exp = exp.(-network.β .* local_energy(network, v))
    projs = projectors(network, v)
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
    h = connecting_tensor(network, (i, j), (i+1, j))
    @cast A[_, u, _, d] := h[u, d]
    A
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T},
    v::Tuple{Rational{Int}, Int},
    ::Val{:central_v}
) where {S, T}
    r, j = v
    i = floor(Int, r)
    u, d = size(interaction_energy(network, (i, j), (i+1, j)))
    (1, u, 1, d)
end


function tensor(
    network::AbstractGibbsNetwork{S, T},
    w::Tuple{Int, Rational{Int}},
    ::Val{:central_h}
) where {S, T}
    i, r = w
    j = floor(Int, r)
    v = connecting_tensor(network, (i, j), (i, j+1))
    @cast A[l, _, r, _] := v[l, r]
    A
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T},
    w::Tuple{Int, Rational{Int}},
    ::Val{:central_h}
) where {S, T}
    i, r = w
    j = floor(Int, r)
    l, r = size(interaction_energy(network, (i, j), (i, j+1)))
    (l, 1, r, 1)
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
    @cast A[_, (u, ũ), _, (d, d̃)] := NW[u, d] * NE[ũ, d̃] 
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
    (1, u*ũ, 1, d*d̃) 
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
    X = network.gauges[v]
    @cast A[_, u, _, d] := Diagonal(X)[u, d]
    A
end


function tensor_size(
    network::AbstractGibbsNetwork{S, T}, 
    v::R,
    ::Val{:gauge_h}
) where {S, T, R}
    X = network.gauges[v]
    u = size(X, 1)
    (1, u, 1, u)
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
    network::FusedNetwork,
    v::Tuple{Int, Int},
    ld::Int,
    l::Int,
    d::Int,
    u::Int
)
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
    network::PEPSNetwork,
    v::Tuple{Int, Int},
    ::Val{:reduced}
)
    i, j = v
    pr = projector(network, v, (i, j+1))
    pd = projector(network, v, (i+1, j))
    @assert size(pr, 1) == size(pr, 1)
    (size(pr, 2), size(pd, 2), size(pd, 1)) 
end 


function mpo(::Type{T},
    peps::AbstractGibbsNetwork,
    r::Union{Rational{Int}, Int}
) where {T <: Number}
    W = MPO(T, length(peps.columns_MPO) * peps.ncols)
    layers = Iterators.product(peps.columns_MPO, 1:peps.ncols)
    for (k, (d, j)) ∈ enumerate(layers) W[k] = tensor(peps, (r, j + d)) end
    W 
end


@memoize Dict mpo(
    peps::AbstractGibbsNetwork,
    r::Union{Rational{Int}, Int}
) = mpo(Float64, peps, r)


@memoize Dict function mps(
    peps::AbstractGibbsNetwork,
    i::Int
) 
    if i > peps.nrows return IdentityMPS() end
    ψ = mps(peps, i+1)
    for r ∈ peps.layers_MPS ψ = mpo(peps, i+r) * ψ end
    compress(ψ, peps)
end


@memoize Dict function dressed_mps(
    peps::AbstractGibbsNetwork,
    i::Int
)
    ψ = mps(peps, i+1)
    for r ∈ peps.layers_left_env ψ = mpo(peps, i+r) * ψ end
    ψ
end


function compress(ψ::AbstractMPS, peps::AbstractGibbsNetwork)
    if bond_dimension(ψ) < peps.bond_dim return ψ end
    SpinGlassTensors.compress(ψ, peps.bond_dim, peps.var_tol, peps.sweeps)
end


@memoize Dict function right_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int}) 
    l = length(∂v)
    if l == 0 return ones(1, 1) end
    R̃ = right_env(peps, i, ∂v[2:l])
    ϕ = dressed_mps(peps, i)
    layers = i .+ reverse(peps.layers_right_env)
    W = prod(mpo.(Ref(peps), layers))
    k = length(W)
    m = ∂v[1]
    M = ϕ[k-l+1]
    M̃ = W[k-l+1]

    K = @view M̃[:, m, :, :]
    @tensor R[x, y] := K[y, β, γ] * M[x, γ, α] * R̃[α, β] order = (β, γ, α)
    R
end


@memoize Dict function left_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int})
    l = length(∂v)
    if l == 0 return [1.] end
    L̃ = left_env(peps, i, ∂v[1:l-1])
    ϕ = dressed_mps(peps, i)
    m = ∂v[l]
    M = ϕ[l]
    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    L
end

