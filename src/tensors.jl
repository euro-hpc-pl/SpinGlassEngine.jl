export tensor

function tensor(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real)
    if v ∉ keys(network.tensors_map) return ones(1, 1) end
    tensor(network, v, β, Val(network.tensors_map[v]))
end

function Base.size(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode)
    if v ∉ keys(network.tensors_map) return (1, 1) end
    size(network, v, Val(network.tensors_map[v]))
end

function tensor(
    network::PEPSNetwork{T, Dense}, v::PEPSNode, β::Real, ::Val{:site}
) where T <: AbstractGeometry
    en = local_energy(network, Node(v))
    loc_exp = exp.(-β .* (en .- minimum(en)))
    projs = projectors(network, Node(v))
    A = zeros(maximum.(projs))
    for (σ, lexp) ∈ enumerate(loc_exp) A[getindex.(projs, Ref(σ))...] += lexp end
    A
end

function tensor(
    network::PEPSNetwork{T, Sparse}, v::PEPSNode, β::Real, ::Val{:sparse_site}
) where T <: AbstractGeometry
    en = local_energy(network, Node(v))
    SparseSiteTensor(
        exp.(-β .* (en .- minimum(en))),
        projectors(network, Node(v))
    )
end

function Base.size(
    network::PEPSNetwork{T, Dense}, v::PEPSNode, ::Val{:site}
) where T <: AbstractGeometry
    maximum.(projectors(network, Node(v)))
end

function Base.size(
    network::PEPSNetwork{T, Sparse}, v::PEPSNode, ::Val{:sparse_site}
) where T <: AbstractGeometry
    size(network, v, Val(:site))
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode, β::Real, ::Val{:central_v}
)
    i = floor(Int, node.i)
    connecting_tensor(network, (i, node.j), (i+1, node.j), β)
end

function Base.size(
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

function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, w::PEPSNode, ::Val{:central_h}
)
    j = floor(Int, node.j)
    size(interaction_energy(network, (node.i, j), (node.i, j+1)))
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, S}, node::PEPSNode, β::Real, ::Val{:central_d}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1), β)
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j), β)
    @cast A[(u, ũ), (d, d̃)] := NW[u, d] * NE[ũ, d̃]
    A
end

function Base.size(
    network::PEPSNetwork{SquareStar{T}, S}, node::PEPSNode, ::Val{:central_d}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (u * ũ, d * d̃)
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, Dense}, node::PEPSNode, β::Real, ::Val{:virtual}
) where T <: AbstractTensorsLayout
    i, j = node.i, floor(Int, node.j)

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
    i, j = node.i, floor(Int, node.j)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    v = Node(node)
    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

    SparseVirtualTensor(h, (vec(p_lb), vec(p_l), vec(p_lt), vec(p_rb), vec(p_r), vec(p_rt)))
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
    Diagonal(network.gauges.data[v])
end

function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:gauge_h}
)
    u = size(network.gauges.data[v], 1)
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

function Base.size(
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

function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:sqrt_down}
)
    r, j = Node(v)
    i = floor(Int, r)
    u, d = size(interaction_energy(network, (i, j), (i+1, j)))
    (min(u, d), d)
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_up_d}
)
    U, Σ, V = svd(tensor(network, v, β, Val(:central_d)))
    U * Diagonal(sqrt.(Σ))
end

function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:sqrt_up_d}
)
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (u * ũ, min(u * ũ, d * d̃))
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_down_d}
)
    U, Σ, V = svd(tensor(network, v, β, Val(:central_d)))
    Diagonal(sqrt.(Σ)) * V'
end

function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:sqrt_down_d}
)
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (min(u * ũ, d * d̃), d * d̃)
end

#-------------- Pegasus --------------

function tensor(
    network::PEPSNetwork{Pegasus, T}, node::PEPSNode, β::Real, ::Val{:pegasus_site}
) where T <: AbstractSparsity
    i, j = node.i, node.j
    j1, j2 = 2*j-1, 2*j

    en1 = local_energy(network, (i, j1))
    en2 = local_energy(network, (i, j2))
    en12 = interaction_energy(network, (i, j1), (i, j2))
    eloc = zeros(length(en1), length(en2))
    p1 = projector(network, (i, j1), (i, j2))
    p2 = projector(network, (i, j2), (i, j1))

    for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
        eloc[s2, s1] = en1[s1] + en2[s2] + en12[p1[s1], p2[s2]]
    end
    eloc = eloc .- minimum(eloc)

    e1d = interaction_energy(network, (i, j1), (i+1, j1))
    e2d = interaction_energy(network, (i, j2), (i+1, j1))
    e1r = interaction_energy(network, (i, j1), (i, j1+2))
    e2r = interaction_energy(network, (i, j2), (i, j1+2))


    # projs = projectors(network, Node(node))
    # A = zeros(maximum.(projs))

    # left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    # prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    # p_lb, p_l, p_lt = last(fuse_projectors(prl))

    # for σ1 ∈ 1:length(en1), σ2 ∈ 1:length(en2)
    #     A[projs] = 0

    # end


    # for (σ, lexp) ∈ enumerate(loc_exp) A[getindex.(projs, Ref(σ))...] += lexp end
    # A
end

# function tensor(
#     network::PEPSNetwork{Pegasus, T}, node::PEPSNode, β::Real, ::Val{:sparse_pegasus_site}
# ) where T <: AbstractSparsity
#     ## TO BE ADDED
# end

function Base.size(
    network::PEPSNetwork{Pegasus, T}, node::PEPSNode, ::Val{:pegasus_site}
) where T <: AbstractSparsity
    maximum.(projectors(network, Node(node)))
end

function Base.size(
    network::PEPSNetwork{Pegasus, T}, node::PEPSNode, ::Val{:sparse_pegasus_site}
) where T <: AbstractSparsity
    maximum.(projectors(network, Node(node)))
end
