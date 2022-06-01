export PegasusSquare

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
        add_edge!(lg, (i, j, 2), (i+1, j, 1))
    end

    for i ∈ 1:m, j ∈ 1:n-1
        add_edge!(lg, (i, j, 2), (i, j+1, 2))
        add_edge!(lg, (i, j, 1), (i, j+1, 2))
    end

    # diagonals
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j, 2), (i+1, j+1, 1))
        add_edge!(lg, (i, j, 1), (i+1, j+1, 2))
    end
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

    if k == 1
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

        for s2 ∈ 1:length(en[2])
            @inbounds ten3[pr[s2], :] += ten2[s2, :]
        end

        RT = R * ten3
        bnd_exp = dropdims(sum(LM[pd[:], :] .* RT[:, p12[:]]', dims=2), dims=2)
        en_min = minimum(en[1])
        loc_exp = exp.(-β .* (en[1] .- en_min))
    else  # k == 2
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
    M = tensor(network, node, β, Val(:pegasus_square_site))

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

    projs = [pl, pu, pr, pd]
    bnd_exp = [le1l, le2l, le1u, le2u]
    bnd_projs = [p1l, p2l, p1u, p2u]
    loc_en = [en1, en2]
    SparsePegasusSquareTensor(M, projs, loc_exp, bnd_exp, bnd_projs, loc_en)
end


function Base.size(M::SparsePegasusSquareTensor, n::Int)
    maximum(M.projs[n])
end


function tensor(
    network::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, β::Real, ::Val{:pegasus_square_site}
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
    network::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, ::Val{:pegasus_square_site}
) where T <: AbstractSparsity
    maximum.(projectors_site_tensor(network, Node(node)))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    network::PEPSNetwork{PegasusSquare, T}, node::PEPSNode, ::Val{:sparse_pegasus_square_site}
) where T <: AbstractSparsity
    maximum.(projectors_site_tensor(network, Node(node)))
end
