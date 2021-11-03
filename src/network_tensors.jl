export
    tensor_size,
    tensor


# tensors signatures are mess    
function tensor(
    network::AbstractGibbsNetwork{S, T},
    v::R,
    β::Real
) where {S, T, R}
    if v ∈ keys(network.tensors_map)
        tensor(network, v, β, Val(network.tensors_map[v]))
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
    network::AbstractGibbsNetwork{Node, T}, 
    v::S,
    β::Real,
    ::Val{:site}
) where {S, T}
    loc_exp = exp.(-β .* local_energy(network, v))
    projs = projectors(network, v)
    # tu ma byc decode projector
    @cast A[σ, _] := loc_exp[σ]
    for pv ∈ projs @cast A[σ, (c, γ)] |= A[σ, c] * pv[σ, γ] end
    B = dropdims(sum(A, dims=1), dims=1)
    reshape(B, size.(projs, 2))
end
 

function tensor_size(
    network::AbstractGibbsNetwork{Node, T}, 
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
    network::AbstractGibbsNetwork{Node, T},
    v::Tuple{Rational{Int}, Int},
    β::Real,
    ::Val{:central_v}
) where T
    r, j = v
    i = floor(Int, r)
    connecting_tensor(network, (i, j), (i+1, j), β)
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, T},
    v::Tuple{Rational{Int}, Int},
    ::Val{:central_v}
) where T
    r, j = v
    i = floor(Int, r)
    size(interaction_energy(network, (i, j), (i+1, j)))
end


function tensor(
    network::AbstractGibbsNetwork{Node, T},
    w::Tuple{Int, Rational{Int}},
    β::Real,
    ::Val{:central_h}
) where T
    i, r = w
    j = floor(Int, r)
    connecting_tensor(network, (i, j), (i, j+1), β)
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, T},
    w::Tuple{Int, Rational{Int}},
    ::Val{:central_h}
) where T
    i, r = w
    j = floor(Int, r)
    size(interaction_energy(network, (i, j), (i, j+1)))
end


function tensor(
    network::AbstractGibbsNetwork{Node, T},
    v::Tuple{Rational{Int}, Rational{Int}},
    β::Real,
    ::Val{:central_d}
) where T
    r, s = v
    i = floor(Int, r)
    j = floor(Int, s)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1), β)
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j), β)
    @cast A[(u, ũ), (d, d̃)] := NW[u, d] * NE[ũ, d̃] 
    A
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, T},
    v::Tuple{Rational{Int}, Rational{Int}},
    ::Val{:central_d}
) where T
    r, s = v
    i = floor(Int, r)
    j = floor(Int, s)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    u * ũ, d * d̃
end


function _all_fused_projectors(
    network::AbstractGibbsNetwork{Node, T},
    v::Tuple{Int, Rational{Int}},
) where T
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
    network::AbstractGibbsNetwork{Node, T},
    v::Tuple{Int, Rational{Int}},
    β::Real,
    ::Val{:virtual}
) where T
    p_lb, p_l, p_lt, 
    p_rb, p_r, p_rt = _all_fused_projectors(network, v)

    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

    @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]    
    @cast A[l, (ũ, u), r, (d̃, d)] |= B[l, r] * p_lt[l, u] * p_rb[r, d] * 
                                     p_rt[r, ũ] * p_lb[l, d̃]
    A
end 


function tensor_size(
    network::AbstractGibbsNetwork{Node, T},
    v::Tuple{Int, Rational{Int}},
    ::Val{:virtual}
) where T
    p_lb, p_l, p_lt, 
    p_rb, p_r, p_rt = _all_fused_projectors(network, v)
    (size(p_l, 1), size(p_lt, 2) * size(p_rt, 2),
     size(p_r, 1), size(p_rb, 2) * size(p_lb, 2))
end


function tensor(
    network::AbstractGibbsNetwork{Node, T}, 
    v::S,
    β::Real,
    ::Val{:gauge_h}
) where {S, T}
    Diagonal(network.gauges_data[v])
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, T}, 
    v::S,
    ::Val{:gauge_h}
) where {S, T}
    u = size(network.gauges_data[v], 1)
    u, u
end


function connecting_tensor(
    network::AbstractGibbsNetwork{Node, T},
    v::Node,
    w::Node,
    β::Real,
) where T
    en = interaction_energy(network, v, w)
    exp.(-β .* (en .- minimum(en)))
end 


function tensor_size(
    network::PEPSNetwork{T},
    v::Node,
    ::Val{:reduced}
) where T <: Square

    i, j = v
    pr = projector(network, v, (i, j+1))
    pd = projector(network, v, (i+1, j))
    @assert size(pr, 1) == size(pr, 1)
    size(pr, 2), size(pd, 2), size(pd, 1)
end
