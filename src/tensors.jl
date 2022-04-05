export tensor

"""
$(TYPEDSIGNATURES)
"""
function tensor(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real)
    if v ∉ keys(network.tensors_map) return ones(1, 1) end
    tensor(network, v, β, Val(network.tensors_map[v]))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode)
    if v ∉ keys(network.tensors_map) return (1, 1) end
    size(network, v, Val(network.tensors_map[v]))
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::PEPSNetwork{T, Dense}, v::PEPSNode, β::Real, ::Val{:site}
) where T <: AbstractGeometry
    en = local_energy(network, Node(v))
    loc_exp = exp.(-β .* (en .- minimum(en)))
    projs = projectors_site_tensor(network, Node(v))
    A = zeros(maximum.(projs))
    for (σ, lexp) ∈ enumerate(loc_exp)
        A[getindex.(projs, Ref(σ))...] += lexp
    end
    A
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::PEPSNetwork{T, Sparse}, v::PEPSNode, β::Real, ::Val{:sparse_site}
) where T <: AbstractGeometry
    en = local_energy(network, Node(v))
    SparseSiteTensor(exp.(-β .* (en .- minimum(en))), projectors_site_tensor(network, Node(v)))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::PEPSNetwork{T, Dense}, v::PEPSNode, ::Union{Val{:site}, Val{:sparse_site}}
) where T <: AbstractGeometry
    maximum.(projectors_site_tensor(network, Node(v)))
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, β::Real, ::Val{:central_v}
)
    i = floor(Int, node.i)
    connecting_tensor(net, (i, node.j), (i+1, node.j), β)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:central_v}
)
    i = floor(Int, node.i)
    size(interaction_energy(network, (i, node.j), (i+1, node.j)))
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, β::Real, ::Val{:central_h}
)
    j = floor(Int, node.j)
    connecting_tensor(net, (node.i, j), (node.i, j+1), β)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:central_h}
)
    j = floor(Int, node.j)
    size(interaction_energy(network, (node.i, j), (node.i, j+1)))
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:gauge_h}
)
    Diagonal(network.gauges.data[v])
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:gauge_h}
)
    u = size(network.gauges.data[v], 1)
    (u, u)
end

"""
$(TYPEDSIGNATURES)
"""
function connecting_tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    en = interaction_energy(network, v, w)
    exp.(-β .* (en .- minimum(en)))
end

"""
$(TYPEDSIGNATURES)
"""
function sqrt_tensor_up(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    U, Σ, _ = svd(connecting_tensor(network, v, w, β))
    U * Diagonal(sqrt.(Σ))
end

"""
$(TYPEDSIGNATURES)
"""
function sqrt_tensor_down(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    _, Σ, V = svd(connecting_tensor(network, v, w, β))
    Diagonal(sqrt.(Σ)) * V'
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_up}
)
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_up(network, (i, j), (i+1, j), β)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:sqrt_up}
)
    r, j = Node(v)
    i = floor(Int, r)
    u, d = size(interaction_energy(network, (i, j), (i+1, j)))
    (u, min(d, u))
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_down}
)
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_down(network, (i, j), (i+1, j), β)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:sqrt_down}
)
    r, j = Node(v)
    i = floor(Int, r)
    u, d = size(interaction_energy(network, (i, j), (i+1, j)))
    (min(u, d), d)
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_up_d}
)
    U, Σ, _ = svd(tensor(network, v, β, Val(:central_d)))
    U * Diagonal(sqrt.(Σ))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:sqrt_up_d}
)
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (u * ũ, min(u * ũ, d * d̃))
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_down_d}
)
    _, Σ, V = svd(tensor(network, v, β, Val(:central_d)))
    Diagonal(sqrt.(Σ)) * V'
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:sqrt_down_d}
)
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (min(u * ũ, d * d̃), d * d̃)
end
