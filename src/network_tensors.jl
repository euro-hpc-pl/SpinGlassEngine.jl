export
    tensor_size,
    tensor

   
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    β::Real
) 
    if v ∈ keys(network.tensors_map)
        tensor(network, v, β, Val(network.tensors_map[v]))
    else
        ones(1, 1)
    end
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, 
    v::PEPSNode
)
    if v ∈ keys(network.tensors_map)
        tensor_size(network, v, Val(network.tensors_map[v]))
    else
        (1, 1)
    end
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    β::Real,
    ::Val{:site}
)
    loc_exp = exp.(-β .* local_energy(network, Node(v)))
    projs = projectors(network, Node(v))
    # writing for 4 projs
    A = zeros(maximum.(projs))
    for (σ, lexp) ∈ enumerate(loc_exp)
        A[projs[1][σ], projs[2][σ], projs[3][σ], projs[4][σ]] += lexp  # more elegent solution?
    end
    A
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    β::Real,
    ::Val{:sparse_site}
)
    loc_exp = exp.(-β .* local_energy(network, Node(v)))
    projs = projectors(network, Node(v))
    SparseSiteTensor(loc_exp, projs)
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, 
    v::PEPSNode,
    ::Val{:site}
) 
    maximum.(projectors(network, Node(v)))
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, 
    v::PEPSNode,
    ::Val{:sparse_site}
) 
    maximum.(projectors(network, Node(v)))
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
    β::Real,
    ::Val{:central_v}
) 
    i = floor(Int, node.i)
    connecting_tensor(network, (i, node.j), (i+1, node.j), β)
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
    ::Val{:central_v}
) 
    i = floor(Int, node.i)
    size(interaction_energy(network, (i, node.j), (i+1, node.j)))
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
    β::Real,
    ::Val{:central_h}
) 
    j = floor(Int, node.j)
    connecting_tensor(network, (node.i, j), (node.i, j+1), β)
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    w::PEPSNode,
    ::Val{:central_h}
) 
    j = floor(Int, node.j)
    size(interaction_energy(network, (node.i, j), (node.i, j+1)))
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
    β::Real,
    ::Val{:central_d}
)
    i = floor(Int, node.i)
    j = floor(Int, node.j)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1), β)
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j), β)
    @cast A[(u, ũ), (d, d̃)] := NW[u, d] * NE[ũ, d̃] 
    A
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
    ::Val{:central_d}
) 
    i = floor(Int, node.i)
    j = floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    u * ũ, d * d̃
end


# function _all_fused_projectors(
#     network::AbstractGibbsNetwork{Node, PEPSNode},
#     node::PEPSNode,
# ) 
#     i = node.i
#     j = floor(Int, node.j)

#     left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
#     prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
#     p_lb, p_l, p_lt = last(fuse_projectors(prl))
#     p_lb = decode_projector!(p_lb)
#     p_l = decode_projector!(p_l)
#     p_lt = decode_projector!(p_lt)

#     right_nbrs = ((i+1, j), (i, j), (i-1, j))
#     prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
#     p_rb, p_r, p_rt = last(fuse_projectors(prr))
#     p_rb = decode_projector!(p_rb)
#     p_r = decode_projector!(p_r)
#     p_rt = decode_projector!(p_rt)

#     p_lb, p_l, p_lt, p_rb, p_r, p_rt
# end


# function tensor(
#     network::AbstractGibbsNetwork{Node, PEPSNode},
#     node::PEPSNode,
#     β::Real,
#     ::Val{:virtual}
# )
#     p_lb, p_l, p_lt, 
#     p_rb, p_r, p_rt = _all_fused_projectors(network, node)
#     v = Node(node)
#     h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)
#     @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]    
#     @cast A[l, (ũ, u), r, (d̃, d)] |= B[l, r] * p_lt[l, u] * p_rb[r, d] * 
#                                      p_rt[r, ũ] * p_lb[l, d̃]
#     A
# end 



function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
    β::Real,
    ::Val{:virtual}
)
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

    A = zeros(length(p_l), maximum(p_rt), maximum(p_lt),
              length(p_r), maximum(p_lb), maximum(p_rb))

    for l ∈ 1:length(p_l)
        for r ∈ 1:length(p_r)
            A[l, p_rt[r], p_lt[l], r, p_lb[l], p_rb[r]] = h[p_l[l], p_r[r]]
        end
    end

    @cast AA[l, (ũ, u), r, (d̃, d)]  |= A[l, ũ, u, r, d̃, d]
    AA
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
    β::Real,
    ::Val{:sparse_virtual}
)
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

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    ::Val{:virtual}
) 
    i, s = v
    j = floor(Int, s)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))
    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))
    (length(p_l), maximum(p_lt) * maximum(p_rt),
    length(p_r), maximum(p_rb) * maximum(p_lb))
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, 
    v::PEPSNode,
    β::Real,
    ::Val{:gauge_h}
) 
    Diagonal(network.gauges_data[v])
end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, 
    v::PEPSNode,
    ::Val{:gauge_h}
) 
    u = size(network.gauges_data[v], 1)
    u, u
end


function connecting_tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::Node,
    w::Node,
    β::Real,
) 
    en = interaction_energy(network, v, w)
    exp.(-β .* (en .- minimum(en)))
end 


function sqrt_tensor_up(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::Node,
    w::Node,
    β::Real
)
    E = connecting_tensor(network, v, w, β)
    U, Σ, V = svd(E)
    U .* sqrt.(Σ)
end 

function sqrt_tensor_down(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::Node,
    w::Node,
    β::Real
)
    E = connecting_tensor(network, v, w, β)
    U, Σ, V = svd(E)
    sqrt.(Σ) .* V'
end 

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    β::Real,
    ::Val{:sqrt_up}
)
    #network::PEPSNetwork{T},
    #v::PEPSNode,
    #::Val{:sqrt_up}
#) where T <: Square
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_up(network, (i, j), (i+1, j), β)
end

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    ::Val{:sqrt_up}
)
    #network::PEPSNetwork{T},
    #v::PEPSNode,
    #::Val{:sqrt_up}
#) where T <: Square
    r, j = Node(v)
    i = floor(Int, r)
    size(interaction_energy(network, (i, j), (i+1, j)))
end

function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    β::Real,
    ::Val{:sqrt_down}
)
    #network::PEPSNetwork{T},
    #v::PEPSNode,
    #::Val{:sqrt_down}
#) where T <: Square
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_down(network, (i, j), (i+1, j), β)
end

function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    v::PEPSNode,
    ::Val{:sqrt_down}
)
    #network::PEPSNetwork{T},
    #v::PEPSNode,
    #::Val{:sqrt_down}
#) where T <: Square
    r, j = Node(v)
    i = floor(Int, r)
    size(interaction_energy(network, (i, j), (i+1, j)))
end