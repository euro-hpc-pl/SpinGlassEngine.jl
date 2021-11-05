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
        #floor.(Int, ones(1, 1))
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
    @cast A[σ, _] := loc_exp[σ]
    for pv ∈ projs 
        pv = decode_projector!(pv) # Zamiana 1d pv -> 2d pv
        @cast A[σ, (c, γ)] |= A[σ, c] * pv[σ, γ]
    end
    B = dropdims(sum(A, dims=1), dims=1)
    reshape(B, maximum.(projs))
end



# function tensor(
#     network::AbstractGibbsNetwork{Node, PEPSNode},
#     v::PEPSNode,
#     β::Real,
#     ::Val{:site_sparse}
# )
#     loc_exp = exp.(-β .* local_energy(network, Node(v)))
#     projs = projectors(network, Node(v))
#     SparseSiteTensor(loc_exp, projs)
# end


function tensor_size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, 
    v::PEPSNode,
    ::Val{:site}
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
    network::AbstractGibbsNetwork{Node, PEPSNode}, #, T},
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


function _all_fused_projectors(
    network::AbstractGibbsNetwork{Node, PEPSNode},
    node::PEPSNode,
) 
    i = node.i
    j = floor(Int, node.j)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))
    p_lb = decode_projector!(p_lb)
    p_l = decode_projector!(p_l)
    p_lt = decode_projector!(p_lt)

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))
    p_rb = decode_projector!(p_rb)
    p_r = decode_projector!(p_r)
    p_rt = decode_projector!(p_rt)

    p_lb, p_l, p_lt, p_rb, p_r, p_rt
end


function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, # ,T},
    node::PEPSNode,
    β::Real,
    ::Val{:virtual}
)  # ehere T <: Dense
    p_lb, p_l, p_lt, 
    p_rb, p_r, p_rt = _all_fused_projectors(network, node)

    v = Node(node)
    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

    @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]    
    @cast A[l, (ũ, u), r, (d̃, d)] |= B[l, r] * p_lt[l, u] * p_rb[r, d] * 
                                     p_rt[r, ũ] * p_lb[l, d̃]
    A
end 

# function tensor(
#     network::PEPSNetwork{T, S},
#     node::PEPSNode,
#     β::Real,
#     ::Val{:virtual}
# )  where T <: SquareStar, S <: Dense
#     p_lb, p_l, p_lt, 
#     p_rb, p_r, p_rt = _all_fused_projectors(network, node)

#     v = Node(node)
#     h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

#     @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]    
#     @cast A[l, (ũ, u), r, (d̃, d)] |= B[l, r] * p_lt[l, u] * p_rb[r, d] * 
#                                      p_rt[r, ũ] * p_lb[l, d̃]
#     A
# end 

# function tensor(
#     network::PEPSNetwork{T, S},
#     node::PEPSNode,
#     β::Real,
#     ::Val{:virtual}
# ) where T <: SquareStar, S <: Sparse
#     p_lb, p_l, p_lt, 
#     p_rb, p_r, p_rt = _all_fused_projectors(network, node)

#     v = Node(node)
#     h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

#     @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]    
#     @cast A[l, (ũ, u), r, (d̃, d)] |= B[l, r] * p_lt[l, u] * p_rb[r, d] * 
#                                      p_rt[r, ũ] * p_lb[l, d̃]
#     SparseVirtualTensor(h, p_lb, p_l, p_lt, p_rb, p_r, p_rt)
# end 


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
    (size(p_l, 1), maximum(p_lt) * maximum(p_rt),
    size(p_r, 1), maximum(p_rb) * maximum(p_lb))
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


function tensor_size(
    network::PEPSNetwork{T},
    v::PEPSNode,
    ::Val{:reduced}
) where T <: Square

    i, j = Node(v)
    pr = decode_projector!(projector(network, v, (i, j+1)))
    pd = decode_projector!(projector(network, v, (i+1, j)))
    @assert size(pr, 1) == size(pr, 1)
    size(pr, 2), size(pd, 2), size(pd, 1)
end
