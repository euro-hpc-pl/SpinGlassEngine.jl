export Square2

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


# """
# $(TYPEDSIGNATURES)
# """
# function update_reduced_env_right(    #function from square might be able to handle this
#     K::AbstractArray{Float64, 1},
#     R::AbstractArray{Float64, 2},
#     M::SparsePegasusSquareTensor,  
#     B::AbstractArray{Float64, 3}
# )
#     pr, pd = M.projs
#     p1l, p2l, p1u, p2u = M.bnd_projs
#     lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)
#     loc_exp12 = CUDA.CuArray(M.loc_exp)  # [s1, s2]

#     K_d = CUDA.CuArray(K)
#     R_d = CUDA.CuArray(R)
#     B_d = CUDA.CuArray(B)

#     @tensor REBu[x, y, β] := B_d[x, y, α] * R_d[α, β]
#     REBs = REBu[:, pd, pr]
#     Kleu1 = K_d .* leu1
#     @tensor Ku[u1, u2] := Kleu1[z, u1] * leu2[z, u2]
#     Ks = Ku[p1u, p2u]  # s1 s2
#     RRs = REBs .* reshape(Ks .* loc_exp12, 1, size(pd, 1), size(pr, 1))

#     ip1l = CUDA.CuArray(diagm(ones(Float64, maximum(p1l))))[p1l, :]  # s1 l1
#     ip2l = CUDA.CuArray(diagm(ones(Float64, maximum(p2l))))[p2l, :]  # s2 l2
#     ll = reshape(lel1, size(lel1, 1), size(lel1, 2), 1) .* reshape(lel2, size(lel2, 1), 1, size(lel2, 2)) # pl l1 l2 # 2.** 12x12x6
#     @tensor ret[x, l] := RRs[x, s1, s2] * ip1l[s1, l1] * ip2l[s2, l2] *  ll[l, l1, l2]  order=(s2, s1, l1, l2)

#     out = Array(ret ./ maximum(abs.(ret)))
#     CUDA.unsafe_free!.((lel1, lel2, leu1, leu2, loc_exp12, K_d, R_d, B_d, REBu, REBs, Kleu1, Ku, RRs, ip1l, ip2l, ll, ret))
#     out
# end

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

function tensor(
    network::PEPSNetwork{Square2, T}, node::PEPSNode, β::Real, ::Val{:sparse_site}
) where T <: AbstractSparsity
    i, j = node.i, node.j

    en1 = local_energy(network, (i, j, 1))
    en2 = local_energy(network, (i, j, 2))
    en12 = interaction_energy(network, (i, j, 1), (i, j, 2))
    p1 = projector(network, (i, j, 1), (i, j, 2))
    p2 = projector(network, (i, j, 2), (i, j, 1))
    eloc12 = en12[p1, p2] .+ reshape(en1, :, 1) .+ reshape(en2, 1, :)
    eloc12 = reshape(eloc12, :)
    expeloc12 = exp.(-β .* (eloc12 .- minimum(eloc12)))
    SparseSiteTensor(
        expeloc12, projectors_site_tensor(network, Node(v))
    )
end


function tensor(
    network::PEPSNetwork{Square2, T}, node::PEPSNode, β::Real, ::Val{:central_h}
) where T <: AbstractSparsity

    i, j = node.i, floor(Int, node.j)

    e11 = connecting_tensor(net, (i, j, 1), (i, j+1, 1), β)
    e12 = connecting_tensor(net, (i, j, 1), (i, j+1, 1), β)
    e21 = connecting_tensor(net, (i, j, 1), (i, j+1, 1), β)
    e22 = connecting_tensor(net, (i, j, 1), (i, j+1, 1), β)

    pl11 = projector(network, (i, j, 1), (i, j+1 ,1))
    pl12 = projector(network, (i, j, 1), (i, j+1 ,2))
    pl21 = projector(network, (i, j, 2), (i, j+1 ,1))
    pl22 = projector(network, (i, j, 2), (i, j+1 ,2))

    pr11 = projector(network, (i, j+1, 1), (i, j ,1))
    pr12 = projector(network, (i, j+1, 1), (i, j ,2))
    pr21 = projector(network, (i, j+1, 2), (i, j ,1))
    pr22 = projector(network, (i, j+1, 2), (i, j ,2))

    SparseCentralTensor(e11, e12, e21, e22, [pl11, pl12, pl21, pl22, pr11, pr12, pr21, pr22])
end


function tensor(
    network::PEPSNetwork{Square2, T}, node::PEPSNode, β::Real, ::Val{:central_v}
) where T <: AbstractSparsity

    i, j = floor(Int, node.i), node.j

    e11 = connecting_tensor(net, (i, j, 1), (i+1, j, 1), β)
    e12 = connecting_tensor(net, (i, j, 1), (i+1, j, 2), β)
    e21 = connecting_tensor(net, (i, j, 2), (i+1, j, 1), β)
    e22 = connecting_tensor(net, (i, j, 2), (i+1, j, 2), β)

    pd11 = projector(network, (i, j, 1), (i+1, j ,1))
    pd12 = projector(network, (i, j, 1), (i+1, j ,2))
    pd21 = projector(network, (i, j, 2), (i+1, j ,1))
    pd22 = projector(network, (i, j, 2), (i+1, j ,2))

    pu11 = projector(network, (i+1, j, 1), (i, j ,1))
    pu12 = projector(network, (i+1, j, 1), (i, j ,2))
    pu21 = projector(network, (i+1, j, 2), (i, j ,1))
    pu22 = projector(network, (i+1, j, 2), (i, j ,2))
    SparseCentralTensor(e11, e12, e21, e22, [pd11, pd12, pd21, pd22, pu11, pu12, pu21, pu22])
end

"""
$(TYPEDSIGNATURES)
"""
function projectors_site_tensor(net::PEPSNetwork{T, S}, vertex::Node) where {T <: Square2, S}
    i, j = vertex
    (
        outer_projector(projector(net, (i, j, 1), ((i, j-1, 1), (i, j-1, 2))),  # l
                        projector(net, (i, j, 2), ((i, j-1, 1), (i, j-1, 2)))),  # l
        outer_projector(projector(net, (i, j, 1), ((i-1, j, 1), (i-1, j, 2))),  # t
                        projector(net, (i, j, 2), ((i-1, j, 1), (i-1, j, 2)))),  # t
        outer_projector(projector(net, (i, j, 1), ((i, j+1, 1), (i, j+1, 2))),  # r
                        projector(net, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))),  # r
        outer_projector(projector(net, (i, j, 1), ((i+1, j, 1), (i+1, j, 2))),  # b
                        projector(net, (i, j, 2), ((i+1, j, 1), (i+1, j, 2)))),  # b
    )
end
