export tensor_size, tensor

function tensor(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real)
    if v ∈ keys(network.tensors_map)
        tensor(network, v, β, Val(network.tensors_map[v]))
    else
        ones(1, 1)
    end
end

function tensor_size(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode)
    if v ∈ keys(network.tensors_map)
        tensor_size(network, v, Val(network.tensors_map[v]))
    else
        (1, 1)
    end
end

function tensor(
    network::PEPSNetwork{T, Dense}, v::PEPSNode, β::Real, ::Val{:site}
) where T <: AbstractGeometry
    loc_exp = exp.(-β .* local_energy(network, Node(v)))
    projs = projectors(network, Node(v))
    A = zeros(maximum.(projs))
    for (σ, lexp) ∈ enumerate(loc_exp) A[getindex.(projs, Ref(σ))...] += lexp end
    A
end

function tensor(
    network::PEPSNetwork{T, Sparse}, v::PEPSNode, β::Real, ::Val{:sparse_site}
) where T <: AbstractGeometry
    SparseSiteTensor(
        exp.(-β .* local_energy(network, Node(v))),
        projectors(network, Node(v))
    )
end

function tensor_size(
    network::PEPSNetwork{T, Dense}, v::PEPSNode, ::Val{:site}
) where T <: AbstractGeometry
    maximum.(projectors(network, Node(v)))
end

function tensor_size(
    network::PEPSNetwork{T, Sparse}, v::PEPSNode, ::Val{:sparse_site}
) where T <: AbstractGeometry
    tensor_size(network, v, Val(:site))
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode, β::Real, ::Val{:central_v}
)
    i = floor(Int, node.i)
    connecting_tensor(network, (i, node.j), (i+1, node.j), β)
end

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode, ::Val{:central_v}
)
    i = floor(Int, node.i)
    size(interaction_energy(network, (i, node.j), (i+1, node.j)))
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode, β::Real, ::Val{:central_h}
)
    j = floor(Int, node.j)
    connecting_tensor(network, (node.i, j), (node.i, j+1), β)
end

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, w::PEPSNode, ::Val{:central_h}
)
    j = floor(Int, node.j)
    size(interaction_energy(network, (node.i, j), (node.i, j+1)))
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, S}, node::PEPSNode, β::Real, ::Val{:central_d}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i = floor(Int, node.i)
    j = floor(Int, node.j)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1), β)
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j), β)
    @cast A[(u, ũ), (d, d̃)] := NW[u, d] * NE[ũ, d̃]
    A
end

function tensor_size(
    network::PEPSNetwork{SquareStar{T}, S}, node::PEPSNode, ::Val{:central_d}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i = floor(Int, node.i)
    j = floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (u * ũ, d * d̃)
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, Dense}, node::PEPSNode, β::Real, ::Val{:virtual}
) where T <: AbstractTensorsLayout
    i = node.i
    j = floor(Int, node.j)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    v = Node(node)
    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

    A = zeros(
            length(p_l), maximum(p_rt), maximum(p_lt),
            length(p_r), maximum(p_lb), maximum(p_rb)
        )

    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        A[l, p_rt[r], p_lt[l], r, p_lb[l], p_rb[r]] = h[p_l[l], p_r[r]]
    end

    @cast AA[l, (ũ, u), r, (d̃, d)] := A[l, ũ, u, r, d̃, d]
    AA
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, Sparse},
    node::PEPSNode, β::Real, ::Val{:sparse_virtual}
) where T <: AbstractTensorsLayout
    i = node.i
    j = floor(Int, node.j)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    v = Node(node)
    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

    SparseVirtualTensor(h, (p_lb, p_l, p_lt, p_rb, p_r, p_rt))
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, Dense}, node::PEPSNode, ::Val{:virtual}
) where T <: AbstractTensorsLayout
    i, s = v
    j = floor(Int, s)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    (length(p_l), maximum(p_lt) * maximum(p_rt), length(p_r), maximum(p_rb) * maximum(p_lb))
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:gauge_h}
)
    Diagonal(network.gauges_data[v])
end

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:gauge_h}
)
    u = size(network.gauges_data[v], 1)
    (u, u)
end

function connecting_tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    en = interaction_energy(network, v, w)
    exp.(-β .* (en .- minimum(en)))
end

function sqrt_tensor_up(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    U, Σ, V = svd(connecting_tensor(network, v, w, β))
    U * Diagonal(sqrt.(Σ))
end

function sqrt_tensor_down(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    U, Σ, V = svd(connecting_tensor(network, v, w, β))
    Diagonal(sqrt.(Σ)) * V'
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_up}
)
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_up(network, (i, j), (i+1, j), β)
end

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:sqrt_up}
)
    r, j = Node(v)
    i = floor(Int, r)
    u, d = size(interaction_energy(network, (i, j), (i+1, j)))
    (u, min(d, u))
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_down}
)
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_down(network, (i, j), (i+1, j), β)
end

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:sqrt_down}
)
    r, j = Node(v)
    i = floor(Int, r)
    u, d = size(interaction_energy(network, (i, j), (i+1, j)))
    (min(u, d), d)
end
