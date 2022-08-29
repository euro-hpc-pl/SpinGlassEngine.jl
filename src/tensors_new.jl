export
    Tensor,
    local_exponent

abstract type AbstractTensor end
abstract type AbstractTensorType end

struct Tensor{T <: AbstractTensorType} <: AbstractTensor
    node::PEPSNode
    network::PEPSNetwork{S, PEPSNode} where S
    array::Array{<:Real}
end

function tensor(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real)
    if v ∉ keys(network.tensors_map) return ones(1, 1) end
    tensor(network, v, β, Val(network.tensors_map[v]))
end

#=
"""
$(TYPEDSIGNATURES)

"""
# needs to be cleaned up
function SparseTensor{Site}(
    net::PEPSNetwork{T, S}, v::PEPSNode, β::Real#, ::Val{:sparse_site}
) where {T <: AbstractGeometry, S}
    SparseSiteTensor(
        local_exponent(local_energy(net, Node(v)), β),
        projectors_site_tensor(net, Node(v))
    )
end
=#

"""
$(TYPEDSIGNATURES)

"""
function Tensor{Site}(
    net::PEPSNetwork{T, Dense}, v::PEPSNode, β::Real
) where T <: AbstractGeometry
    sp = SparseTensor{Site}(net, v, β)
    A = zeros(maximum.(sp.projs))
    for (σ, lexp) ∈ enumerate(sp.loc_exp)
        @inbounds A[getindex.(sp.projs, Ref(σ))...] += lexp
    end
   Tensor{Site}(v, net, A)
end

"""
$(TYPEDSIGNATURES)

"""
@inline function Base.size(A::Tensor{Site})
    maximum.(projectors_site_tensor(A.network, Node(A.node)))
end

"""
$(TYPEDSIGNATURES)

"""
function Tensor{CentralVertical}(
    net::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, β::Real
)
    i = floor(Int, node.i)
    A = connecting_tensor(net, (i, node.j), (i+1, node.j), β)
    Tensor{CentralVertical}(node, net, A)
end

"""
$(TYPEDSIGNATURES)

"""
@inline function Base.size(A::Tensor{CentralVertical})
    i = floor(Int, A.node.i)
    size(interaction_energy(A.network, (i, A.node.j), (i+1, A.node.j)))
end

"""
$(TYPEDSIGNATURES)

"""
function Tesnor{CentralHorizontal}(
    net::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, β::Real
)
    j = floor(Int, node.j)
    A = connecting_tensor(net, (node.i, j), (node.i, j+1), β)
    Tesnor{CentralHorizontal}(v, net, A)
end

"""
$(TYPEDSIGNATURES)

"""
@inline function Base.size(A::Tesnor{CentralHorizontal})
    j = floor(Int, A.node.j)
    size(interaction_energy(A.network, (A.node.i, j), (A.node.i, j+1)))
end

"""
$(TYPEDSIGNATURES)

"""
function Tensor{GaugeHorizontal}(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real
)
    Diagonal(network.gauges.data[v])
end

"""
$(TYPEDSIGNATURES)

"""
@inline function Base.size(A::Tensor{GaugeHorizontal})
    u = size(A.network.gauges.data[v], 1)
    (u, u)
end

"""
$(TYPEDSIGNATURES)

"""
@inline function local_exponent(en::T, β::Real) where T <: AbstractArray
    en_min = minimum(en)
    exp.(-β .* (en .- en_min))
end

"""
$(TYPEDSIGNATURES)

"""
@inline function connecting_tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    local_exponent(interaction_energy(net, v, w), β)
end

"""
$(TYPEDSIGNATURES)

"""
function Tensor{SqrtUp}(
    net::AbstractGibbsNetwork{Node, T}, v::T, β::Real
)
    r, j = Node(v)
    i = floor(Int, r)
    U, Σ, _ = svd(connecting_tensor(net, (i, j), (i+1, j), β))
    A = U * Diagonal(sqrt.(Σ))
    Tensor{SqrtUp}(v, net, A)
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(A::Tensor{SqrtUp})
    r, j = Node(A.node)
    i = floor(Int, r)
    u, d = size(interaction_energy(A.network, (i, j), (i+1, j)))
    (u, min(d, u))
end

"""
$(TYPEDSIGNATURES)

"""
function Tensor{SqrtDown}(
    net::AbstractGibbsNetwork{Node, T}, v::T, β::Real
) where T <: PEPSNode
    r, j = Node(v)
    i = floor(Int, r)
    _, Σ, V = svd(connecting_tensor(net, (i, j), (i+1, j), β))
    A = Diagonal(sqrt.(Σ)) * V'
    Tensor{SqrtDown}(v, net, A)
end


"""
$(TYPEDSIGNATURES)

"""
function Base.size(A::Tensor{SqrtDown})
    r, j = Node(A.node)
    i = floor(Int, r)
    u, d = size(interaction_energy(A.network, (i, j), (i+1, j)))
    (min(u, d), d)
end

"""
$(TYPEDSIGNATURES)

"""
function Tensor{SqrtUpDiagonal}(
    network::AbstractGibbsNetwork{Node, T}, v::T, β::Real
) where T <: PEPSNode
    U, Σ, _ = svd(Tensor{CentralDiagonal}(network, v, β))
    A = U * Diagonal(sqrt.(Σ))
    A = Tensor{SqrtUpDiagonal}(v, net, A)
end


"""
$(TYPEDSIGNATURES)

"""
function Base.size(A::Tensor{SqrtUpDiagonal})
    i, j = floor(Int, A.node.i), floor(Int, A.node.j)
    u, d = size(interaction_energy(A.network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(A.network, (i, j + 1), (i + 1, j)))
    (u * ũ, min(u * ũ, d * d̃))
end

"""
$(TYPEDSIGNATURES)

"""
function Tensor{SqrtDownDiagonal}(
    network::AbstractGibbsNetwork{Node, T}, v::T, β::Real
) where T <: PEPSNode
    _, Σ, V = svd(tensor(network, v, β, Val(:central_d)))
    A = Diagonal(sqrt.(Σ)) * V'
    Tensor{SqrtDownDiagonal}(v, net, A)
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(A::Tensor{SqrtDownDiagonal})
    i, j = floor(Int, A.node.i), floor(Int, A.node.j)
    u, d = size(interaction_energy(A.network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(A.network, (i, j + 1), (i + 1, j)))
    (min(u * ũ, d * d̃), d * d̃)
end
