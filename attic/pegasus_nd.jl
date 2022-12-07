export
    PegasusSquare,
    update_reduced_env_right

"""
$(TYPEDSIGNATURES)
"""
struct PegasusSquare <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)
"""
function PegasusSquare(m::Int, n::Int)
    labels = [(i, j, k) for j ∈ 1:n for i ∈ 1:m for k ∈ 1:2]
    lg = LabelledGraph(labels)
    for i ∈ 1:m, j ∈ 1:n add_edge!(lg, (i, j, 1), (i, j, 2)) end

    for i ∈ 1:m-1, j ∈ 1:n
        add_edge!(lg, (i, j, 1), (i+1, j, 1))
        add_edge!(lg, (i, j, 1), (i+1, j, 2))
    end

    for i ∈ 1:m, j ∈ 1:n-1
        add_edge!(lg, (i, j, 2), (i, j+1, 2))
        add_edge!(lg, (i, j, 2), (i, j+1, 1))
    end

    # diagonals
    # for i ∈ 1:m-1, j ∈ 1:n-1
    #     add_edge!(lg, (i, j, 2), (i+1, j+1, 1))
    #     add_edge!(lg, (i, j, 1), (i+1, j+1, 2))
    # end
    lg
end

"""
$(TYPEDSIGNATURES)

Geometry: 2 nodes -> 1 TN site. This will work for Chimera.
"""
pegasus_square_site(::Type{Dense}) = :pegasus_square_site

"""
$(TYPEDSIGNATURES)
"""
pegasus_square_site(::Type{Sparse}) = :sparse_pegasus_square_site

"""
$(TYPEDSIGNATURES)
"""
function tensor_map(
    ::Type{PegasusSquare}, ::Type{S}, nrows::Int, ncols::Int
) where S <: AbstractSparsity
    map = Dict{PEPSNode, Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols push!(map, PEPSNode(i, j) => pegasus_square_site(S)) end
    map
end

"""
$(TYPEDSIGNATURES)
"""
function gauges_list(::Type{PegasusSquare}, nrows::Int, ncols::Int)
    [
        GaugeInfo(
            (PEPSNode(i + 1//3, j), PEPSNode(i + 2//3, j)),
            PEPSNode(i , j),
            4,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: PegasusSquare
    MpoLayers(
        Dict(i => (-1//3, 0, 1//3) for i ∈ 1:ncols),
        Dict(i => (1//3,) for i ∈ 1:ncols),
        Dict(i => (0,) for i ∈ 1:ncols)
    )
end

"""
$(TYPEDSIGNATURES)
"""
function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, ∂v::Vector{Int}
) where {T <: PegasusSquare, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j, m = ctr.current_node
    @nexprs 2 m->(v_m = (i, j, m))

    L = left_env(ctr, i, ∂v[1:j-1], indβ)
    R = right_env(ctr, i, ∂v[(j+4):end], indβ)
    M = dressed_mps(ctr, i, indβ)[j]

    @tensor LM[y, z] := L[x] * M[x, y, z]

    if m == 1
        # here one has to avarage over s2
        @nexprs 2 k->(
            el_k = projected_energy(ctr.peps, v_k, (i, j-1, 2), ∂v[j-1+k]);
            eu_k = projected_energy(ctr.peps, v_k, (i-1, j, 1), ∂v[j+1+k]);
            en_k = local_energy(ctr.peps, v_k) .+ el_k .+ eu_k
        )
        en21 = interaction_energy(ctr.peps, v_2, v_1)
        p21 = projector(ctr.peps, v_2, v_1)
        p12 = projector(ctr.peps, v_1, v_2)

        pr = projector(ctr.peps, (i, j, 2), @ntuple 2 k->(i, j+1, k))
        pd = projector(ctr.peps, (i, j, 1), @ntuple 2 k->(i+1, j, k))

        ten = reshape(en_2, :, 1) .+ en21[p21, :]
        RT = R[:, pr] * exp.(-β .* (ten .- minimum(ten)))

        bnd_exp = dropdims(sum(LM[pd, :] .* RT[:, p12]', dims=2), dims=2)
        loc_exp = exp.(-β .* (en_1 .- minimum(en_1)))
    else
        # m == 2; here s1 is fixed
        eng_l = projected_energy(ctr.peps, v_2, (i, j-1, 2), ∂v[j])
        eng_u = projected_energy(ctr.peps, v_2, (i-1, j, 1), ∂v[j+3])
        eng_21 = projected_energy(ctr.peps, v_2, (i, j, 1), ∂v[j+2])

        en = local_energy(ctr.peps, v_2) .+ eng_l .+ eng_u .+ eng_21
        loc_exp = exp.(-β .* (en .- minimum(en)))

        lmx = @inbounds LM[∂v[j+1], :]
        @tensor lmxR[y] := lmx[x] * R[x, y]

        pr = projector(ctr.peps, v_2, @ntuple 2 k->(i, j+1, k))
        bnd_exp = lmxR[pr]
    end

    probs = loc_exp .* bnd_exp
    push!(ctr.statistics, ((i, j, m), ∂v) => error_measure(probs))
    normalize_probability(probs)
end

"""
$(TYPEDSIGNATURES)
"""
function update_reduced_env_right(
    K::AbstractArray{Float64, 1},
    R::AbstractArray{Float64, 2},
    M::SparsePegasusSquareTensor,
    B::AbstractArray{Float64, 3}
)
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    ip1l = cuIdentity(eltype(K), maximum(p1l))[p1l, :]
    ip2l = cuIdentity(eltype(K), maximum(p2l))[p2l, :]

    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)
    K_d, R_d, B_d = CUDA.CuArray.((K, R, B))

    @tensor REBu[x, y, β] := B_d[x, y, α] * R_d[α, β]
    @tensor Ku[u1, u2] := (K_d .* leu1)[z, u1] * leu2[z, u2]

    REBs = REBu[:, pd, pr]
    Kexp = Ku[p1u, p2u] .* CUDA.CuArray(M.loc_exp)

    @cast RRs[x, y, z] := REBs[x, y, z] * Kexp[y, z]
    @cast ll[x, y, z] := lel1[x, y] * lel2[x, z]
    @tensor ret[x, l] := RRs[x, s1, s2] * ip1l[s1, l1] * ip2l[s2, l2] * ll[l, l1, l2] order=(s2, s1, l1, l2)

    Array(ret ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function nodes_search_order_Mps(peps::PEPSNetwork{T, S}) where {T <: PegasusSquare, S}
    ([(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2], (peps.nrows+1, 1, 1))
end

function _boundary(i::T, j::T, ::Val{1}, N::T) where T<:Int
    vcat(
        (((i, m, 1), @ntuple 2 k->(i+1, m, k)) for m ∈ 1:j-1)...,
        (@ntuple 2 k->(i, j+k-2, 3-k)),
        (@ntuple 2 k->(i, j+k-2, 2)),
        (@ntuple 2 k->(i+k-2, j, 1)),
        (@ntuple 2 k->(i+k-2, j, k)),
        (((i-1, m, 1), @ntuple 2 k->(i, m, k)) for m ∈ j+1:N)...
    )
end

function _boundary(i::T, j::T, ::Val{2}, N::T) where T<:Int
    vcat(
        (((i, m, 1), @ntuple 2 k->(i+1, m, k)) for m ∈ 1:j-1)...,
        (@ntuple 2 k->(i, j+k-2, 2)),
        ((i, j, 1), @ntuple 2 k->(i+1, j, k)),
        (@ntuple 2 k->(i, j, k)),
        (@ntuple 2 k->(i+k-2, j, k)),
        (((i-1, m, 1), @ntuple 2 k->(i, m, k)) for m ∈ j+1:N)...
    )
end

"""
$(TYPEDSIGNATURES)
"""
function boundary(::Type{T}, ctr::MpsContractor{S}, node::Node) where {T <: PegasusSquare, S}
    i, j, k = node
    _boundary(i, j, Val(k), ctr.peps.ncols)
end

"""
$(TYPEDSIGNATURES)
"""
function update_energy(
    ::Type{T}, ctr::MpsContractor{S}, σ::Vector{Int}) where {T <: PegasusSquare, S}
    net = ctr.peps
    i, j, k = ctr.current_node
    en = local_energy(net, (i, j, k))
    for v ∈ ((i, j-1, 2), (i-1, j, 1))
        en += bond_energy(net, (i, j, k), v, local_state_for_node(ctr, σ, v))
    end
    if k != 2 return en end
    en += bond_energy(net, (i, j, k), (i, j, 1), local_state_for_node(ctr, σ, (i, j, 1)))
    en
end

"""
$(TYPEDSIGNATURES)

cluster-cluster energies attached from left and top
"""
function tensor(
    net::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, β::Real, ::Val{:sparse_pegasus_square_site}
) where T <: AbstractSparsity
    i, j = node.i, node.j
    @nexprs 2 k->(v_k = (i, j, k))

    pr = projector(net, v_2, @ntuple 2 k->(i, j+1, k))
    pd = projector(net, v_1, @ntuple 2 k->(i+1, j, k))

    pl, (pl_1, pl_2) = fuse_projectors(
        @ntuple 2 k->projector(net, (i, j-1, 2), v_k)
    )
    pu, (pu_1, pu_2) = fuse_projectors(
        @ntuple 2 k->projector(net, (i-1, j, 1), v_k)
    )
    @nexprs 2 k->(
        eu_k = interaction_energy(net, (i-1, j, 1), v_k)[pu_k, :];
        el_k = interaction_energy(net, (i, j-1, 2), v_k)[pl_k, :];

        el_k = probability(el_k, β);
        eu_k = probability(eu_k, β);

        pl_k = projector(net, v_k, (i, j-1 ,2));
        pu_k = projector(net, v_k, (i-1, j, 1));

        en_k = local_energy(net, v_k);
    )
    en12 = projected_energy(net, v_1, v_2)
    @cast eloc[x, y] := en12[x, y] + en_1[x] + en_2[y]

    SparsePegasusSquareTensor(
        [pr, pd],
        probability(eloc, β),
        [el_1, el_2, eu_1, eu_2],
        [pl_1, pl_2, pu_1, pu_2],
        maximum.((pl, pu, pr, pd))
    )
end

Base.size(M::SparsePegasusSquareTensor, n::Int) = M.sizes[n]
Base.size(M::SparsePegasusSquareTensor) = M.sizes


"""
$(TYPEDSIGNATURES)
 This code could be simplified ....
"""
function tensor(
    network::PEPSNetwork{PegasusSquare, Dense}, node::PEPSNode, β::Real, ::Val{:pegasus_square_site}
) where T <: AbstractSparsity

    i, j = node.i, node.j

    en1 = local_energy(network, (i, j, 1))
    en2 = local_energy(network, (i, j, 2))
    en12 = interaction_energy(network, (i, j, 1), (i, j, 2))

    eloc = zeros(length(en2), length(en1))
    p1 = projector(network, (i, j, 1), (i, j, 2))
    p2 = projector(network, (i, j, 2), (i, j, 1))

    for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
        @inbounds eloc[s2, s1] = en1[s1] + en2[s2] + en12[p1[s1], p2[s2]]
    end
    eloc = eloc .- minimum(eloc)
    loc_exp = exp.(-β .* eloc)

    pr = projector(network, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
    pd = projector(network, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))

    p1l = projector(network, (i, j, 1), (i, j-1 ,2))
    p2l = projector(network, (i, j, 2), (i, j-1, 2))
    p1u = projector(network, (i, j, 1), (i-1, j, 1))
    p2u = projector(network, (i, j, 2), (i-1, j, 1))

    pl1 = projector(network, (i, j-1, 2), (i, j, 1))
    pl2 = projector(network, (i, j-1, 2), (i, j, 2))
    pl, (pl1, pl2) = fuse_projectors((pl1, pl2))

    pu1 = projector(network, (i-1, j, 1), (i, j, 1))
    pu2 = projector(network, (i-1, j, 1), (i, j, 2))
    pu, (pu1, pu2) = fuse_projectors((pu1, pu2))

    e1u = interaction_energy(network, (i, j, 1), (i-1, j, 1))
    e2u = interaction_energy(network, (i, j, 2), (i-1, j, 1))
    e1l = interaction_energy(network, (i, j, 1), (i, j-1, 2))
    e2l = interaction_energy(network, (i, j, 2), (i, j-1, 2))

    e1u = @inbounds @view e1u[:, pu1]
    e2u = @inbounds @view e2u[:, pu2]
    e1l = @inbounds @view e1l[:, pl1]
    e2l = @inbounds @view e2l[:, pl2]

    le1u = exp.(-β .* (e1u .- minimum(e1u)))
    le2u = exp.(-β .* (e2u .- minimum(e2u)))
    le1l = exp.(-β .* (e1l .- minimum(e1l)))
    le2l = exp.(-β .* (e2l .- minimum(e2l)))

        A = zeros(eltype(β), maximum.((pl, pu, pr, pd)))
    for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
        @inbounds ll = reshape(le1l[p1l[s1], :], :, 1) .* reshape(le2l[p2l[s2], :], :, 1)
        @inbounds lu = reshape(le1u[p1u[s1], :], 1, :) .* reshape(le2u[p2u[s2], :], 1, :)
        @inbounds A[:, :, pr[s2], pd[s1]] += loc_exp[s2, s1] .* (ll .* lu)
    end
    A ./ maximum(A)s
end


"""
$(TYPEDSIGNATURES)
"""
function projectors_site_tensor(net::PEPSNetwork{T, S}, vertex::Node) where {T <: PegasusSquare, S}
    i, j = vertex
    (
        projector(net, (i, j-1, 2), @ntuple 2 k->(i, j, k)),
        projector(net, (i-1, j, 1), @ntuple 2 k->(i, j, k)),
        projector(net, (i, j, 2), @ntuple 2 k->(i, j+1, k)),
        projector(net, (i, j, 1), @ntuple 2 k->(i+1, j, k))
    )
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    net::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, ::Val{:pegasus_square_site}
) where T <: AbstractSparsity
    maximum.(projectors_site_tensor(net, Node(node)))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    net::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, ::Val{:sparse_pegasus_square_site}
) where T <: AbstractSparsity
    maximum.(projectors_site_tensor(net, Node(node)))
end
