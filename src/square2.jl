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
site2(::Type{Dense}) = :site_square2

"""
$(TYPEDSIGNATURES)
"""
site2(::Type{Sparse}) = :sparse_site_square2

"""
$(TYPEDSIGNATURES)
"""
function tensor_map(
    ::Type{Square2{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{GaugesEnergy, EnergyGauges}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site2(S))
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
    #main = Dict{Site, Sites}(i => (-1//6, 0, 3//6, 4//6) for i ∈ 1:ncols)
    main = Dict{Site, Sites}(i => (0, 3//6) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-3//6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    #MpoLayers(main, Dict(i => (3//6, 4//6) for i ∈ 1:ncols), right)
    MpoLayers(main, Dict(i => (3//6,) for i ∈ 1:ncols), right)

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

# """   # TODO
# $(TYPEDSIGNATURES)
# """
# function conditional_probability(   # TODO
#     ::Type{T}, ctr::MpsContractor{S}, ∂v::Vector{Int}
# ) where {T <: Square2, S}
#     indβ, β = length(ctr.betas), last(ctr.betas)
#     i, j, k = ctr.current_node

#     L = left_env(ctr, i, ∂v[1:j-1], indβ)
#     M = dressed_mps(ctr, i, indβ)[j]
#     @tensor LM[y, z] := L[x] * M[x, y, z]

#     if k == 1  # here has to avarage over s2
#         R = right_env(ctr, i, ∂v[(j+8):end], indβ)

#         eng_loc = [local_energy(ctr.peps, (i, j, k)) for k ∈ 1:2]
#         el = [interaction_energy(ctr.peps, (i, j, k), (i, j-1, m)) for k ∈ 1:2, m ∈ 1:2]
#         pl = [projector(ctr.peps, (i, j, k), (i, j-1, m)) for k ∈ 1:2, m ∈ 1:2]
#         eng_l = [@view el[k, m][pl[k, m][:], ∂v[j - 1 + k + (m - 1) * 2]] for k ∈ 1:2, m ∈ 1:2]

#         eu = [interaction_energy(ctr.peps, (i, j, k), (i-1, j, m)) for k ∈ 1:2, m ∈ 1:2]
#         pu = [projector(ctr.peps, (i, j, k), (i-1, j, m)) for k ∈ 1:2, m ∈ 1:2]
#         eng_u = [@view eu[k][pu[k][:], ∂v[j + 3 + k + (m - 1) * 2]] for k ∈ 1:2, m ∈ 1:2]

#         en = [eng_loc[k] .+ eng_l[k, 1] .+ eng_l[k, 2] .+ eng_u[k, 1] .+ eng_u[k, 2] for k ∈ 1:2]

#         en21 = interaction_energy(ctr.peps, (i, j, 2), (i, j, 1))
#         p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
#         p12 = projector(ctr.peps, (i, j, 1), (i, j, 2))

#         pr = projector(ctr.peps, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
#         pd = projector(ctr.peps, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))
        
#         # pr = outer_projector(
#         #         projector(ctr.peps, (i, j, 1), ((i, j+1, 1), (i, j+1, 2))),  # r
#         #         projector(ctr.peps, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))))
#         # pd = outer_projector(
#         #         projector(ctr.peps, (i, j, 1), ((i+1, j, 1), (i+1, j, 2))),  # b
#         #         projector(ctr.peps, (i, j, 2), ((i+1, j, 1), (i+1, j, 2))))

#         ten = reshape(en[2], (:, 1)) .+ en21[p21[:], :]
#         ten_min = minimum(ten)
#         ten2 = exp.(-β .* (ten .- ten_min))

#         #ten3 = zeros(maximum(pr), size(ten2, 2))
#         # for s2 ∈ 1:length(en[2])
#         #     @inbounds ten3[pr[s2], :] += ten2[s2, :]
#         # end
#         RT = R[:, pr] * ten2 # R * ten3
#         bnd_exp = dropdims(sum(LM[pd[:], :] .* RT[:, p12[:]]', dims=2), dims=2)
#         en_min = minimum(en[1])
#         loc_exp = exp.(-β .* (en[1] .- en_min))
#     else  # k == 2 ; here s1 is fixed
#         R = right_env(ctr, i, ∂v[(j+4):end], indβ)
#         eng_loc = local_energy(ctr.peps, (i, j, 2))

#         el = interaction_energy(ctr.peps, (i, j, 2), (i, j-1, 2))
#         pl = projector(ctr.peps, (i, j, 2), (i, j-1, 2))
#         eng_l = @inbounds @view el[pl[:], ∂v[j]]

#         eu = interaction_energy(ctr.peps, (i, j, 2), (i-1, j, 1))
#         pu = projector(ctr.peps, (i, j, 2), (i-1, j, 1))
#         eng_u = @inbounds @view eu[pu[:], ∂v[j+3]]

#         e21 = interaction_energy(ctr.peps, (i, j, 2), (i, j, 1))
#         p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
#         eng_21 = @inbounds @view e21[p21[:], ∂v[j+2]]

#         en = eng_loc .+ eng_l .+ eng_u .+ eng_21
#         en_min = minimum(en)
#         loc_exp = exp.(-β .* (en .- en_min))

#         pr = projector(ctr.peps, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
#         lmx = @inbounds @view LM[∂v[j+1], :]

#         @tensor lmxR[y] := lmx[x] * R[x, y]
#         @inbounds bnd_exp = lmxR[pr[:]]
#     end

#     probs = loc_exp .* bnd_exp
#     push!(ctr.statistics, ((i, j, k), ∂v) => error_measure(probs))
#     normalize_probability(probs)
# end



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
function boundary(::Type{T}, ctr::MpsContractor{S}, node::Node) where {T <: Square2, S}  # TODO
    i, j, k = node
    if k == 1
        bnd = vcat(
            [((i, m, 1), ((i+1, m, 1), (i+1, m, 2)), (i, m, 2), ((i+1, m, 1), (i+1, m, 2))) for m ∈ 1:j-1]...,
            ((i, j-1, 2), (i, j, 1)),
            ((i, j-1, 2), (i, j, 2)),
            ((i-1, j, 1), (i, j, 1)),
            ((i-1, j, 1), (i, j, 2)),
            # ((i, j-1, 1), (i, j, 1)),
            # ((i, j-1, 1), (i, j, 2)),
            # ((i, j-1, 2), (i, j, 1)),
            # ((i, j-1, 2), (i, j, 2)),
            # ((i-1, j, 1), (i, j, 1)),
            # ((i-1, j, 1), (i, j, 2)),
            # ((i-1, j, 2), (i, j, 1)),
            # ((i-1, j, 2), (i, j, 2)),
            [((i-1, m, 1), ((i, m, 1), (i, m, 2)), (i-1, m, 2), ((i, m, 1), (i, m, 2))) for m ∈ j+1:ctr.peps.ncols]...
        )
    else  # k == 2
        bnd = vcat(
            [((i, m, 1), ((i+1, m, 1), (i+1, m, 2)), (i, m, 2), ((i+1, m, 1), (i+1, m, 2))) for m ∈ 1:j-1]...,
            ((i, j-1, 2), (i, j, 2)),
            ((i, j, 1), ((i+1, j, 1), (i+1, j, 2))),
            ((i, j, 1), (i, j, 2)),
            ((i-1, j, 1), (i, j, 2)),
            [((i-1, m, 1), ((i, m, 1), (i, m, 2)), (i-1, m, 2), ((i, m, 1), (i, m, 2))) for m ∈ j+1:ctr.peps.ncols]...
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
    for v ∈ ((i, j-1, 1), (i, j-1, 2), (i-1, j, 1), (i-1, j, 2))
        en += bond_energy(net, (i, j, k), v, local_state_for_node(ctr, σ, v))
    end
    if k != 2 return en end
    en += bond_energy(net, (i, j, k), (i, j, 1), local_state_for_node(ctr, σ, (i, j, 1)))  # here k=2
    en
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{Square2{S}, T}, node::PEPSNode, β::Real, ::Val{:sparse_site_square2}
) where {T <: AbstractSparsity, S <: AbstractTensorsLayout}
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
    net::PEPSNetwork{Square2{S}, T}, node::PEPSNode, β::Real, ::Val{:site_square2}
) where {T <: AbstractSparsity, S <: AbstractTensorsLayout}
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
    network::PEPSNetwork{Square2{S}, T}, node::PEPSNode, β::Real, ::Val{:central_h}
) where {T <: AbstractSparsity, S <: AbstractTensorsLayout}
    i, j = node.i, floor(Int, node.j)

    p11r = projector(network, (i, j+1, 1), (i, j, 1))
    p21r = projector(network, (i, j+1, 1), (i, j, 2))
    p12r = projector(network, (i, j+1, 2), (i, j ,1))
    p22r = projector(network, (i, j+1, 2), (i, j, 2))

    p1r, (p11r, p21r) = fuse_projectors((p11r, p21r))
    p2r, (p12r, p22r) = fuse_projectors((p12r, p22r))

    p21l = projector(network, (i, j, 2), (i, j+1, 1))
    p22l = projector(network, (i, j, 2), (i, j+1, 2))
    p12l = projector(network, (i, j, 1), (i, j+1, 2))
    p11l = projector(network, (i, j, 1), (i, j+1, 1))

    p1l, (p11l, p12l) = fuse_projectors((p11l, p12l))
    p2l, (p21l, p22l) = fuse_projectors((p21l, p22l))

    e11 = interaction_energy(network, (i, j, 1), (i, j+1, 1))
    e12 = interaction_energy(network, (i, j, 1), (i, j+1, 2))
    e21 = interaction_energy(network, (i, j, 2), (i, j+1, 1))
    e22 = interaction_energy(network, (i, j, 2), (i, j+1, 2))

    e11 = e11[p11l, p11r]
    e21 = e21[p21l, p21r]
    e12 = e12[p12l, p12r]
    e22 = e22[p22l, p22r]

    le11 = exp.(-β .* (e11 .- minimum(e11)))
    le21 = exp.(-β .* (e21 .- minimum(e21)))
    le12 = exp.(-β .* (e12 .- minimum(e12)))
    le22 = exp.(-β .* (e22 .- minimum(e22)))

    sl = maximum(outer_projector(
        projector(network, (i, j, 1), ((i, j+1, 1), (i, j+1, 2))),  # r
        projector(network, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))),  # r
    ))

    sr = maximum(outer_projector(
            projector(network, (i, j+1, 1), ((i, j, 1), (i, j, 2))),  # l
            projector(network, (i, j+1, 2), ((i, j, 1), (i, j, 2))),  # l
        ))
    SparseCentralTensor(
        le11,
        le12,
        le21,
        le22,
        (sl, sr)
    )
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::PEPSNetwork{Square2{S}, Dense}, node::PEPSNode, β::Real, ::Val{:central_h}
) where S <: AbstractTensorsLayout
    i, j = node.i, floor(Int, node.j)

    p11r = projector(network, (i, j+1, 1), (i, j, 1))
    p21r = projector(network, (i, j+1, 1), (i, j, 2))
    p12r = projector(network, (i, j+1, 2), (i, j ,1))
    p22r = projector(network, (i, j+1, 2), (i, j, 2))

    p1r, (p11r, p21r) = fuse_projectors((p11r, p21r))
    p2r, (p12r, p22r) = fuse_projectors((p12r, p22r))

    p21l = projector(network, (i, j, 2), (i, j+1, 1))
    p22l = projector(network, (i, j, 2), (i, j+1, 2))
    p12l = projector(network, (i, j, 1), (i, j+1, 2))
    p11l = projector(network, (i, j, 1), (i, j+1, 1))

    p1l, (p11l, p12l) = fuse_projectors((p11l, p12l))
    p2l, (p21l, p22l) = fuse_projectors((p21l, p22l))

    e11 = interaction_energy(network, (i, j, 1), (i, j+1, 1))
    e12 = interaction_energy(network, (i, j, 1), (i, j+1, 2))
    e21 = interaction_energy(network, (i, j, 2), (i, j+1, 1))
    e22 = interaction_energy(network, (i, j, 2), (i, j+1, 2))

    e11 = e11[p11l, p11r]
    e21 = e21[p21l, p21r]
    e12 = e12[p12l, p12r]
    e22 = e22[p22l, p22r]

    le11 = exp.(-β .* (e11 .- minimum(e11)))
    le21 = exp.(-β .* (e21 .- minimum(e21)))
    le12 = exp.(-β .* (e12 .- minimum(e12)))
    le22 = exp.(-β .* (e22 .- minimum(e22)))

    @cast V[(l1, l2), (r1, r2)] := le11[l1,r1] * le21[l2, r1] * le12[l1, r2] * le22[l2, r2]
    V ./ maximum(V)
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::PEPSNetwork{Square2{S}, T}, node::PEPSNode, β::Real, ::Val{:central_v}
) where {T <: AbstractSparsity, S <: AbstractTensorsLayout}
    i, j = floor(Int, node.i), node.j

    p11u = projector(network, (i+1, j, 1), (i, j, 1))
    p12u = projector(network, (i+1, j, 2), (i, j, 1))
    p21u = projector(network, (i+1, j, 1), (i, j, 2))
    p22u = projector(network, (i+1, j, 2), (i, j, 2))

    p1u, (p11u, p21u) = fuse_projectors((p11u, p21u))
    p2u, (p12u, p22u) = fuse_projectors((p12u, p22u))

    p11d = projector(network, (i, j, 1), (i+1, j, 1))
    p12d = projector(network, (i, j, 1), (i+1, j, 2))
    p21d = projector(network, (i, j, 2), (i+1, j, 1))
    p22d = projector(network, (i, j, 2), (i+1, j, 2))

    p1d, (p11d, p12d) = fuse_projectors((p11d, p12d))
    p2d, (p21d, p22d) = fuse_projectors((p21d, p22d))

    e11 = interaction_energy(network, (i, j, 1), (i+1, j, 1))
    e21 = interaction_energy(network, (i, j, 2), (i+1, j, 1))
    e12 = interaction_energy(network, (i, j, 1), (i+1, j, 2))
    e22 = interaction_energy(network, (i, j, 2), (i+1, j, 2))

    e11 = e11[p11d, p11u]
    e21 = e21[p21d, p21u]
    e12 = e12[p12d, p12u]
    e22 = e22[p22d, p22u]

    le11 = exp.(-β .* (e11 .- minimum(e11)))
    le21 = exp.(-β .* (e21 .- minimum(e21)))
    le12 = exp.(-β .* (e12 .- minimum(e12)))
    le22 = exp.(-β .* (e22 .- minimum(e22)))

    su = maximum(outer_projector(
        projector(network, (i, j, 1), ((i+1, j, 1), (i+1, j, 2))),  # b
        projector(network, (i, j, 2), ((i+1, j, 1), (i+1, j, 2))),  # b
    ))
    sd = maximum(outer_projector(
        projector(network, (i+1, j, 1), ((i, j, 1), (i, j, 2))),  # t
        projector(network, (i+1, j, 2), ((i, j, 1), (i, j, 2))),  # t
    ))

    SparseCentralTensor(
        le11,
        le12,
        le21,
        le22,
        (su, sd)
    )
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    network::PEPSNetwork{Square2{S}, Dense}, node::PEPSNode, β::Real, ::Val{:central_v}
) where S <: AbstractTensorsLayout
    i, j = floor(Int, node.i), node.j

    p11u = projector(network, (i+1, j, 1), (i, j, 1))
    p12u = projector(network, (i+1, j, 2), (i, j, 1))
    p21u = projector(network, (i+1, j, 1), (i, j, 2))
    p22u = projector(network, (i+1, j, 2), (i, j, 2))

    p1u, (p11u, p21u) = fuse_projectors((p11u, p21u))
    p2u, (p12u, p22u) = fuse_projectors((p12u, p22u))

    p11d = projector(network, (i, j, 1), (i+1, j, 1))
    p12d = projector(network, (i, j, 1), (i+1, j, 2))
    p21d = projector(network, (i, j, 2), (i+1, j, 1))
    p22d = projector(network, (i, j, 2), (i+1, j, 2))

    p1d, (p11d, p12d) = fuse_projectors((p11d, p12d))
    p2d, (p21d, p22d) = fuse_projectors((p21d, p22d))

    e11 = interaction_energy(network, (i, j, 1), (i+1, j, 1))
    e21 = interaction_energy(network, (i, j, 2), (i+1, j, 1))
    e12 = interaction_energy(network, (i, j, 1), (i+1, j, 2))
    e22 = interaction_energy(network, (i, j, 2), (i+1, j, 2))

    e11 = e11[p11d, p11u]
    e21 = e21[p21d, p21u]
    e12 = e12[p12d, p12u]
    e22 = e22[p22d, p22u]

    le11 = exp.(-β .* (e11 .- minimum(e11)))
    le21 = exp.(-β .* (e21 .- minimum(e21)))
    le12 = exp.(-β .* (e12 .- minimum(e12)))
    le22 = exp.(-β .* (e22 .- minimum(e22)))

    @cast V[(u1, u2), (d1, d2)] := le11[u1, d1] * le21[u2, d1] * le12[u1, d2] * le22[u2, d2]

    V ./ maximum(V)
end

Base.size(M::SparseCentralTensor, n::Int) = M.sizes[n]
Base.size(M::SparseCentralTensor) = M.sizes

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
