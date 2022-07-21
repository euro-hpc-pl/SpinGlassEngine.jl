export Square2, tensor

"""
$(TYPEDSIGNATURES)
"""
struct Square2{T <: AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)
"""
function Square2(m::Int, n::Int)
    labels = [(i, j, k) for j ∈ 1:n for i ∈ 1:m for k ∈ 1:2]
    lg = LabelledGraph(labels)
    for i ∈ 1:m, j ∈ 1:n add_edge!(lg, (i, j, 1), (i, j, 2)) end

    for i ∈ 1:m-1, j ∈ 1:n
        add_edge!(lg, (i, j, 1), (i+1, j, 1))
        add_edge!(lg, (i, j, 1), (i+1, j, 2))
        add_edge!(lg, (i, j, 2), (i+1, j, 1))
        add_edge!(lg, (i, j, 2), (i+1, j, 2))
    end

    for i ∈ 1:m, j ∈ 1:n-1
        add_edge!(lg, (i, j, 2), (i, j+1, 2))
        add_edge!(lg, (i, j, 2), (i, j+1, 1))
        add_edge!(lg, (i, j, 1), (i, j+1, 2))
        add_edge!(lg, (i, j, 1), (i, j+1, 1))
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
"""
function tensor_map(
    ::Type{Square2{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{GaugesEnergy, EnergyGauges}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site(S))
        if j < ncols push!(map, PEPSNode(i, j + 1//2) => :central_h) end
        if i < nrows push!(map, PEPSNode(i + 1//2, j) => :central_v) end
    end
    map
end


"""
$(TYPEDSIGNATURES)

Assigns gauges and corresponding information to GaugeInfo structure for a given Layout.
"""
function gauges_list(::Type{Square2{T}}, nrows::Int, ncols::Int) where T <: GaugesEnergy
    [
        GaugeInfo(
            (PEPSNode(i + 1//6, j), PEPSNode(i + 2//6, j)),
            PEPSNode(i + 1//2, j),
            1,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

Assigns gauges and corresponding information to GaugeInfo structure for a given Layout.
"""
function gauges_list(::Type{Square2{T}}, nrows::Int, ncols::Int) where T <: EnergyGauges
    [
        GaugeInfo(
            (PEPSNode(i + 4//6, j), PEPSNode(i + 5//6, j)),
            PEPSNode(i + 1//2, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the Square geometry with the EnergyGauges layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: Square2{EnergyGauges}
    main = Dict{Site, Sites}(i => (-1//6, 0, 3//6, 4//6) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-3//6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (3//6, 4//6) for i ∈ 1:ncols), right)
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the Square geometry with the GaugesEnergy layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: Square2{GaugesEnergy}
    main = Dict{Site, Sites}(i => (-4//6, -1//2, 0, 1//6) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-3//6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (1//6,) for i ∈ 1:ncols), right)
end

"""
$(TYPEDSIGNATURES)
"""
function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, ∂v::Vector{Int}
) where {T <: Square2, S}
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
function nodes_search_order_Mps(peps::PEPSNetwork{T, S}) where {T <: Square2, S}
    ([(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2], (peps.nrows+1, 1, 1))
end

"""
$(TYPEDSIGNATURES)
"""
function boundary(::Type{T}, ctr::MpsContractor{S}, node::Node) where {T <: Square2, S}
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
    ::Type{T}, ctr::MpsContractor{S}, σ::Vector{Int}) where {T <: Square2, S}
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
"""
function tensor(
    net::PEPSNetwork{Square2, T}, node::PEPSNode, β::Real, ::Val{:sparse_site}
) where T <: AbstractSparsity
    i, j = node.i, node.j
    en1 = local_energy(net, (i, j, 1))
    en2 = local_energy(net, (i, j, 2))
    en12 = interaction_energy(net, (i, j, 1), (i, j, 2))

    p1 = projector(net, (i, j, 1), (i, j, 2))
    p2 = projector(net, (i, j, 2), (i, j, 1))

    eloc12 = reshape(en12[p1, p2] .+ reshape(en1, :, 1) .+ reshape(en2, 1, :), :)
    mloc = minimum(eloc12)

    SparseSiteTensor(
        exp.(-β .* (eloc12 .- mloc)),
        projectors_site_tensor(net, Node(node))
    )
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{Square2, T}, node::PEPSNode, β::Real, ::Val{:site}
) where T <: AbstractSparsity
    i, j = node.i, node.j
    en1 = local_energy(net, (i, j, 1))
    en2 = local_energy(net, (i, j, 2))
    en12 = interaction_energy(net, (i, j, 1), (i, j, 2))

    p1 = projector(net, (i, j, 1), (i, j, 2))
    p2 = projector(net, (i, j, 2), (i, j, 1))

    eloc12 = reshape(en12[p1, p2] .+ reshape(en1, :, 1) .+ reshape(en2, 1, :), :)
    mloc = minimum(eloc12)

    loc_exp = exp.(-β .* (eloc12 .- mloc))
    projs = projectors_site_tensor(net, Node(node))
    A = zeros(maximum.(projs))
    for (σ, lexp) ∈ enumerate(loc_exp)
        @inbounds A[getindex.(projs, Ref(σ))...] += lexp
    end
    A
end


"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{Square2, T}, node::PEPSNode, β::Real, ::Val{:central_h}
) where T <: AbstractSparsity
    i, j = node.i, floor(Int, node.j)

    SparseCentralTensor(
        connecting_tensor(net, (i, j, 1), (i, j+1, 1), β),
        connecting_tensor(net, (i, j, 1), (i, j+1, 1), β),
        connecting_tensor(net, (i, j, 1), (i, j+1, 1), β),
        connecting_tensor(net, (i, j, 1), (i, j+1, 1), β),
        [
            projector(net, (i, j, 1), (i, j+1 ,1)),
            projector(net, (i, j, 1), (i, j+1 ,2)),
            projector(net, (i, j, 2), (i, j+1 ,1)),
            projector(net, (i, j, 2), (i, j+1 ,2)),
            projector(net, (i, j+1, 1), (i, j ,1)),
            projector(net, (i, j+1, 1), (i, j ,2)),
            projector(net, (i, j+1, 2), (i, j ,1)),
            projector(net, (i, j+1, 2), (i, j ,2))
        ]
    )
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{Square2, T}, node::PEPSNode, β::Real, ::Val{:central_v}
) where T <: AbstractSparsity
    i, j = floor(Int, node.i), node.j

    SparseCentralTensor(
        connecting_tensor(net, (i, j, 1), (i+1, j, 1), β),
        connecting_tensor(net, (i, j, 1), (i+1, j, 2), β),
        connecting_tensor(net, (i, j, 2), (i+1, j, 1), β),
        connecting_tensor(net, (i, j, 2), (i+1, j, 2), β),
        [
            projector(net, (i, j, 1), (i+1, j ,1)),
            projector(net, (i, j, 1), (i+1, j ,2)),
            projector(net, (i, j, 2), (i+1, j ,1)),
            projector(net, (i, j, 2), (i+1, j ,2)),
            projector(net, (i+1, j, 1), (i, j ,1)),
            projector(net, (i+1, j, 1), (i, j ,2)),
            projector(net, (i+1, j, 2), (i, j ,1)),
            projector(net, (i+1, j, 2), (i, j ,2))
        ]
    )
end

"""
$(TYPEDSIGNATURES)
"""
function projectors_site_tensor(net::PEPSNetwork{T, S}, vertex::Node) where {T <: Square2, S}
    i, j = vertex
    (
        outer_projector(
            projector(net, (i, j, 1), ((i, j-1, 1), (i, j-1, 2))),  # l
            projector(net, (i, j, 2), ((i, j-1, 1), (i, j-1, 2))),  # l
        ),
        outer_projector(
            projector(net, (i, j, 1), ((i-1, j, 1), (i-1, j, 2))),  # t
            projector(net, (i, j, 2), ((i-1, j, 1), (i-1, j, 2))),  # t
        ),
        outer_projector(
            projector(net, (i, j, 1), ((i, j+1, 1), (i, j+1, 2))),  # r
            projector(net, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))),  # r
        ),
        outer_projector(
            projector(net, (i, j, 1), ((i+1, j, 1), (i+1, j, 2))),  # b
            projector(net, (i, j, 2), ((i+1, j, 1), (i+1, j, 2))),  # b
        ),
    )
end
