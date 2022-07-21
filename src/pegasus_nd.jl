export PegasusSquare, update_reduced_env_right

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
    i, j, k = ctr.current_node

    L = left_env(ctr, i, ∂v[1:j-1], indβ)
    R = right_env(ctr, i, ∂v[(j+4):end], indβ)
    M = dressed_mps(ctr, i, indβ)[j]

    @tensor LM[y, z] := L[x] * M[x, y, z]
    if k == 1  # here has to avarage over s2
        eng_loc = [local_energy(ctr.peps, (i, j, k)) for k ∈ 1:2]
        el = [interaction_energy(ctr.peps, (i, j, k), (i, j-1, 2)) for k ∈ 1:2]
        pl = [projector(ctr.peps, (i, j, k), (i, j-1, 2)) for k ∈ 1:2]
        eng_l = [@view el[k][pl[k][:], ∂v[j-1+k]] for k ∈ 1:2]

        eu = [interaction_energy(ctr.peps, (i, j, k), (i-1, j, 1)) for k ∈ 1:2]
        pu = [projector(ctr.peps, (i, j, k), (i-1, j, 1)) for k ∈ 1:2]
        eng_u = [@view eu[k][pu[k][:], ∂v[j+1+k]] for k ∈ 1:2]

        en = [eng_loc[k] .+ eng_l[k] .+ eng_u[k] for k ∈ 1:2]

        en21 = interaction_energy(ctr.peps, (i, j, 2), (i, j, 1))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
        p12 = projector(ctr.peps, (i, j, 1), (i, j, 2))

        pr = projector(ctr.peps, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
        pd = projector(ctr.peps, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))

        ten = reshape(en[2], (:, 1)) .+ en21[p21[:], :]
        ten_min = minimum(ten)
        ten2 = exp.(-β .* (ten .- ten_min))
        ten3 = zeros(maximum(pr), size(ten2, 2))

        # for s2 ∈ 1:length(en[2])
        #     @inbounds ten3[pr[s2], :] += ten2[s2, :]
        # end

        RT = R[:, pr] * ten2 # R * ten3
        bnd_exp = dropdims(sum(LM[pd[:], :] .* RT[:, p12[:]]', dims=2), dims=2)
        en_min = minimum(en[1])
        loc_exp = exp.(-β .* (en[1] .- en_min))
    else  # k == 2 ; here s1 is fixed
        eng_loc = local_energy(ctr.peps, (i, j, 2))

        el = interaction_energy(ctr.peps, (i, j, 2), (i, j-1, 2))
        pl = projector(ctr.peps, (i, j, 2), (i, j-1, 2))
        eng_l = @inbounds @view el[pl[:], ∂v[j]]

        eu = interaction_energy(ctr.peps, (i, j, 2), (i-1, j, 1))
        pu = projector(ctr.peps, (i, j, 2), (i-1, j, 1))
        eng_u = @inbounds @view eu[pu[:], ∂v[j+3]]

        e21 = interaction_energy(ctr.peps, (i, j, 2), (i, j, 1))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
        eng_21 = @inbounds @view e21[p21[:], ∂v[j+2]]

        en = eng_loc .+ eng_l .+ eng_u .+ eng_21
        en_min = minimum(en)
        loc_exp = exp.(-β .* (en .- en_min))

        pr = projector(ctr.peps, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
        lmx = @inbounds @view LM[∂v[j+1], :]

        @tensor lmxR[y] := lmx[x] * R[x, y]
        @inbounds bnd_exp = lmxR[pr[:]]
    end

    probs = loc_exp .* bnd_exp
    push!(ctr.statistics, ((i, j, k), ∂v) => error_measure(probs))
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

"""
$(TYPEDSIGNATURES)
"""
function boundary(::Type{T}, ctr::MpsContractor{S}, node::Node) where {T <: PegasusSquare, S}
    i, j, k = node
    if k == 1
        bnd = vcat(
            [((i, m, 1), ((i+1, m, 1), (i+1, m, 2))) for m ∈ 1:j-1]...,
            ((i, j-1, 2), (i, j, 1)),
            ((i, j-1, 2), (i, j, 2)),
            ((i-1, j, 1), (i, j, 1)),
            ((i-1, j, 1), (i, j, 2)),
            [((i-1, m, 1), ((i, m, 1), (i, m, 2))) for m ∈ j+1:ctr.peps.ncols]...
        )
    else  # k == 2
        bnd = vcat(
            [((i, m, 1), ((i+1, m, 1), (i+1, m, 2))) for m ∈ 1:j-1]...,
            ((i, j-1, 2), (i, j, 2)),
            ((i, j, 1), ((i+1, j, 1), (i+1, j, 2))),
            ((i, j, 1), (i, j, 2)),
            ((i-1, j, 1), (i, j, 2)),
            [((i-1, m, 1), ((i, m, 1), (i, m, 2))) for m ∈ j+1:ctr.peps.ncols]...
        )
    end
    bnd
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
    network::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, β::Real, ::Val{:sparse_pegasus_square_site}
) where T <: AbstractSparsity
    i, j = node.i, node.j

    en1 = local_energy(network, (i, j, 1))
    en2 = local_energy(network, (i, j, 2))
    en12 = interaction_energy(network, (i, j, 1), (i, j, 2))

    p1 = projector(network, (i, j, 1), (i, j, 2))
    p2 = projector(network, (i, j, 2), (i, j, 1))

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

    eu1 = interaction_energy(network, (i-1, j, 1), (i, j, 1))
    eu2 = interaction_energy(network, (i-1, j, 1), (i, j, 2))
    el1 = interaction_energy(network, (i, j-1, 2), (i, j, 1))
    el2 = interaction_energy(network, (i, j-1, 2), (i, j, 2))

    eu1 = @inbounds eu1[pu1, :]
    eu2 = @inbounds eu2[pu2, :]
    el1 = @inbounds el1[pl1, :]
    el2 = @inbounds el2[pl2, :]

    leu1 = exp.(-β .* (eu1 .- minimum(eu1)))
    leu2 = exp.(-β .* (eu2 .- minimum(eu2)))
    lel1 = exp.(-β .* (el1 .- minimum(el1)))
    lel2 = exp.(-β .* (el2 .- minimum(el2)))

    eloc12 = en12[p1, p2] .+ reshape(en1, :, 1) .+ reshape(en2, 1, :)

    SparsePegasusSquareTensor(
        [pr, pd],
        exp.(-β .* (eloc12 .- minimum(eloc12))),
        [lel1, lel2, leu1, leu2],
        [p1l, p2l, p1u, p2u],
        maximum.((pl, pu, pr, pd))
    )
end

Base.size(M::SparsePegasusSquareTensor, n::Int) = M.sizes[n]
Base.size(M::SparsePegasusSquareTensor) = M.sizes

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, β::Real, ::Val{:pegasus_square_site}
) where T <: AbstractSparsity
    i, j = node.i, node.j

    p1 = projector(net, (i, j, 1), (i, j, 2))
    p2 = projector(net, (i, j, 2), (i, j, 1))

    pr = projector(net, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
    pd = projector(net, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))

    for α ∈ 1:2
        @eval begin
            "en$α" = local_energy(net, (i, j, α))
            "p$(α)l" = projector(net, (i, j, α), (i, j-1 ,2))
            "p$(α)u" = projector(net, (i, j, α), (i-1, j, 1))
            "pl$α" = projector(net, (i, j-1, 2), (i, j, α))
            "pu$α" = projector(net, (i-1, j, 1), (i, j, α))
        end
    end

    for x ∈ (:u, :l)
        t = ("p$(x)1", "p$(x)2")
        @eval "p$x", t = fuse_projectors(t)
    end

    eloc = interaction_energy(net, (i, j, 1), (i, j, 2))[p1, p2]
    for α ∈ 1:2
        @eval begin
            eloc .+= reshape("en$α", :, 1)
            "e$(α)u" = interaction_energy(net, (i, j, α), (i-1, j, 1))
            "e$(α)l" = interaction_energy(net, (i, j, α), (i, j-1, 2))
            for x ∈ (:u, :l)
                e = "e$(α)$(x)"
                e = @inbounds e[:, "p$(x)$(α)"]
                emin = minimum(e)
                "le$(α)$(x)" = exp.(-β .* (e .- emin))
            end
        end
    end

    emin = minimum(eloc)
    loc_exp = exp.(-β .* (eloc' .- emin))

    A = zeros(maximum.((pl, pu, pr, pd)))
    for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
        @inbounds ll = reshape(le1l[p1l[s1], :], :, 1) .* reshape(le2l[p2l[s2], :], :, 1)
        @inbounds lu = reshape(le1u[p1u[s1], :], 1, :) .* reshape(le2u[p2u[s2], :], 1, :)
        @inbounds A[:, :, pr[s2], pd[s1]] += loc_exp[s2, s1] .* (ll .* lu)
    end
    A ./ maximum(A)
end

"""
$(TYPEDSIGNATURES)
"""
function projectors_site_tensor(net::PEPSNetwork{T, S}, vertex::Node) where {T <: PegasusSquare, S}
    i, j = vertex
    (
        projector(net, (i, j-1, 2), ((i, j, 1), (i, j, 2))),
        projector(net, (i-1, j, 1), ((i, j, 1), (i, j, 2))),
        projector(net, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))),
        projector(net, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))
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
