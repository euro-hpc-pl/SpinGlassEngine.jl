export SquareStar2

struct SquareStar2{T <: AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)
"""
function SquareStar2(m::Int, n::Int)
    lg = Square2(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j, 1), (i+1, j+1, 1))
        add_edge!(lg, (i, j, 1), (i+1, j+1, 2))
        add_edge!(lg, (i, j, 2), (i+1, j+1, 1))
        add_edge!(lg, (i, j, 2), (i+1, j+1, 2))
        add_edge!(lg, (i+1, j, 1), (i, j+1, 1))
        add_edge!(lg, (i+1, j, 1), (i, j+1, 2))
        add_edge!(lg, (i+1, j, 2), (i, j+1, 1))
        add_edge!(lg, (i+1, j, 2), (i, j+1, 2))
    end
    lg
end

"""
$(TYPEDSIGNATURES)
"""
Virtual2(::Type{Dense}) = :virtual2

"""
$(TYPEDSIGNATURES)
"""
Virtual2(::Type{Sparse}) = :sparse_virtual2

"""
$(TYPEDSIGNATURES)
"""
function tensor_map(
    ::Type{SquareStar2{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{EnergyGauges, GaugesEnergy}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site2(S),
            PEPSNode(i, j - 1//2) => Virtual2(S),
            PEPSNode(i + 1//2, j) => :central_v2
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1   # why from 0?
        push!(map, PEPSNode(i + 1//2, j + 1//2) => :central_d2)
    end
    map
end


"""
$(TYPEDSIGNATURES)
"""
function gauges_list(::Type{SquareStar2{T}}, nrows::Int, ncols::Int) where T <: GaugesEnergy
    [
        GaugeInfo(
            (PEPSNode(i + 1//6, j), PEPSNode(i + 2//6, j)),
            PEPSNode(i + 1//2, j),
            1,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

"""
$(TYPEDSIGNATURES)
"""
function gauges_list(::Type{SquareStar2{T}}, nrows::Int, ncols::Int) where T <: EnergyGauges
    [
        GaugeInfo(
            (PEPSNode(i + 4//6, j), PEPSNode(i + 5//6, j)),
            PEPSNode(i + 1//2, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end


"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareStar2 geometry with the EnergyGauges layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar2{EnergyGauges}
    MpoLayers(
        Dict(site(i) => (-1//6, 0, 3//6, 4//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (3//6, 4//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareStar2 geometry with the GaugesEnergy layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar2{GaugesEnergy}
    MpoLayers(
        Dict(site(i) => (-4//6, -1//2, 0, 1//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1//6,) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end


@memoize Dict function precompute_conditional(
    ::Type{T}, ctr::MpsContractor{S}, current_node
) where {T <: SquareStar2, S}
    i, j, k = current_node
    β = last(ctr.betas)
    if k == 1
        en12 = interaction_energy(ctr.peps, (i, j, 1), (i, j, 2))
        p12 = projector(ctr.peps, (i, j, 1), (i, j, 2))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))

        plb1 = projector(ctr.peps, (i, j, 1), ((i+1, j-1, 1), (i+1, j-1, 2)))
        plb2 = projector(ctr.peps, (i, j, 2), ((i+1, j-1, 1), (i+1, j-1, 2)))
        prf1 = projector(ctr.peps, (i, j, 1), ((i+1, j+1, 1), (i+1, j+1, 2), (i, j+1, 1), (i, j+1, 2), (i-1, j+1, 1), (i-1, j+1, 2)))
        prf2 = projector(ctr.peps, (i, j, 2), ((i+1, j+1, 1), (i+1, j+1, 2), (i, j+1, 1), (i, j+1, 2), (i-1, j+1, 1), (i-1, j+1, 2)))
        pd1 = projector(ctr.peps, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))
        pd2 = projector(ctr.peps, (i, j, 2), ((i+1, j, 1), (i+1, j, 2)))

        plb = outer_projector(plb1, plb2)
        prf = outer_projector(prf1, prf2)
        pd = outer_projector(pd1, pd2)

        eng_loc = [local_energy(ctr.peps, (i, j, k)) for k ∈ 1:2]

        el = [interaction_energy(ctr.peps, (i, j, k), (i, j-1, m)) for k ∈ 1:2, m ∈ 1:2]
        pl = [projector(ctr.peps, (i, j, k), (i, j-1, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_l = [el[k, m][pl[k, m], :] for k ∈ 1:2, m ∈ 1:2]

        elu = [interaction_energy(ctr.peps, (i, j, k), (i-1, j-1, m)) for k ∈ 1:2, m ∈ 1:2]
        plu = [projector(ctr.peps, (i, j, k), (i-1, j-1, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_lu = [elu[k, m][plu[k, m], :] for k ∈ 1:2, m ∈ 1:2]

        eu = [interaction_energy(ctr.peps, (i, j, k), (i-1, j, m)) for k ∈ 1:2, m ∈ 1:2]
        pu = [projector(ctr.peps, (i, j, k), (i-1, j, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_u = [eu[k, m][pu[k, m], :] for k ∈ 1:2, m ∈ 1:2]

        le = en12[p12, p21]
        ele = exp.(-β .* (le .- minimum(le)))

        splb = maximum(plb)

        if ctr.onGPU
            ele = CuArray(ele)
            eng_loc = [CuArray(eng_loc[k]) for k ∈ 1:2]
            eng_l = [CuArray(eng_l[k, m]) for k ∈ 1:2, m ∈ 1:2]
            eng_lu = [CuArray(eng_lu[k, m]) for k ∈ 1:2, m ∈ 1:2]
            eng_u = [CuArray(eng_u[k, m]) for k ∈ 1:2, m ∈ 1:2]
            plb = CuArray(plb)
            prf = CuArray(prf)
            pd = CuArray(pd)
        end

        return (ele, eng_loc, eng_l, eng_lu, eng_u, plb, prf, pd, splb)
    else
        eng_loc = local_energy(ctr.peps, (i, j, 2))

        el = [interaction_energy(ctr.peps, (i, j, 2), (i, j-1, m)) for m ∈ 1:2]
        pl = [projector(ctr.peps, (i, j, 2), (i, j-1, m)) for m ∈ 1:2]
        eng_l = [el[m][pl[m], :] for m ∈ 1:2]

        elu = [interaction_energy(ctr.peps, (i, j, 2), (i-1, j-1, m)) for m ∈ 1:2]
        plu = [projector(ctr.peps, (i, j, 2), (i-1, j-1, m)) for m ∈ 1:2]
        eng_lu = [elu[m][plu[m], :] for m ∈ 1:2]

        eu = [interaction_energy(ctr.peps, (i, j, 2), (i-1, j, m)) for m ∈ 1:2]
        pu = [projector(ctr.peps, (i, j, 2), (i-1, j, m)) for m ∈ 1:2]
        eng_u = [eu[m][pu[m], :] for m ∈ 1:2]

        en12 = interaction_energy(ctr.peps, (i, j, 1), (i, j, 2))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
        eng_12 = @view en12[:, p21]

        plb1 = projector(ctr.peps, (i, j, 1), ((i+1, j-1, 1), (i+1, j-1, 2)))
        plb2 = projector(ctr.peps, (i, j, 2), ((i+1, j-1, 1), (i+1, j-1, 2)))
        prf2 = projector(ctr.peps, (i, j, 2), ((i+1, j+1, 1), (i+1, j+1, 2), (i, j+1, 1), (i, j+1, 2), (i-1, j+1, 1), (i-1, j+1, 2)))
        pd2 = projector(ctr.peps, (i, j, 2), ((i+1, j, 1), (i+1, j, 2)))

        splb1, splb2, sprf2, spd2 = maximum.((plb1, plb2, prf2, pd2))

        if ctr.onGPU
            eng_loc = CuArray(eng_loc)
            eng_l = [CuArray(eng_l[m]) for m ∈ 1:2]
            eng_lu = [CuArray(eng_lu[m]) for m ∈ 1:2]
            eng_u = [CuArray(eng_u[m]) for m ∈ 1:2]
            eng_12 = CuArray(eng_12)
            plb2 = CuArray(plb2)
            prf2 = CuArray(prf2)
            pd2 = CuArray(pd2)
        end

        return (eng_loc, eng_l, eng_lu, eng_u, eng_12, plb2, prf2, pd2, splb1, splb2, sprf2, spd2)
    end
end




"""
$(TYPEDSIGNATURES)
"""
function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, ∂v::Vector{Int}
) where {T <: SquareStar2, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j, k = ctr.current_node

    L = left_env(ctr, i, ∂v[1:2*j-2], indβ)
    ψ = dressed_mps(ctr, i, indβ)
    MX, M = ψ[j - 1//2], ψ[j]
    @tensor LMX[y, z] := L[x] * MX[x, y, z]

    if k == 1  # here has to avarage over s2
        R = right_env(ctr, i, ∂v[(2 * j + 12) : end], indβ)
        if ctr.onGPU
            R = CuArray(R)
        end
    
        ele, eng_loc, eng_l, eng_lu, eng_u, plb, prf, pd, splb = precompute_conditional(T, ctr, ctr.current_node)

        eng_l = [@view eng_l[k, m][:, ∂v[2 * j - 1 + k + (m - 1) * 2]] for k ∈ 1:2, m ∈ 1:2]
        eng_lu = [@view eng_lu[k, m][:, ∂v[2 * j + 3 + k + (m - 1) * 2]] for k ∈ 1:2, m ∈ 1:2]
        eng_u = [@view eng_u[k, m][:, ∂v[2 * j + 7 + k + (m - 1) * 2]] for k ∈ 1:2, m ∈ 1:2]
        en = [eng_loc[k] .+ eng_l[k, 1] .+ eng_l[k, 2] .+ eng_lu[k, 1] .+ eng_lu[k, 2] .+ eng_u[k, 1] .+ eng_u[k, 2] for k ∈ 1:2]
        en = [exp.(-β .* (en[k] .- minimum(en[k]))) for k ∈ 1:2]
        tele = reshape(en[1], (:, 1)) .* ele .* reshape(en[2], (1, :))

        @cast LMX4[x, y, v, p] := LMX[(x, y), (v, p)]  (x ∈ 1:1, p ∈ 1:splb)
        LMX3 = LMX4[:, :, ∂v[2 * j - 1], :]
        @cast R3[x, y, p] := R[(x, y), p] (y ∈ 1:1)

        from = 1
        total_size = length(pd)

        total_memory = 2^33
        s1, s2, _ = size(M)
        batch_size = max(Int(floor(total_memory / (8 * (s1 * s2 + s1 + s2 + 1)))), 1)
        batch_size = Int(2^floor(log2(batch_size) + 1e-6))
        LR = typeof(M) <: CuArray ? CUDA.zeros(eltype(M), total_size) : zeros(eltype(M), total_size)
        while from <= total_size
            to = min(total_size, from + batch_size - 1)
            vplb = @view plb[from:to]
            vpd = @view pd[from:to]
            vprf = @view prf[from:to]
            @inbounds LMX3p = LMX3[:, :, vplb]
            @inbounds Mp = M[:, :, vpd]
            @inbounds R3p = R3[:, :, vprf]
            @inbounds LR[from:to] = dropdims(LMX3p ⊠ Mp ⊠ R3p, dims=(1, 2))
            from = to + 1
        end
        LR = reshape(LR, size(tele))
        LR .*= tele
        probs = Array(dropdims(sum(LR, dims=2), dims=2))
    else  # k == 2
        R = right_env(ctr, i, ∂v[(2 * j + 10) : end], indβ)
        if ctr.onGPU
            R = CuArray(R)
        end
    
        eng_loc, eng_l, eng_lu, eng_u, eng_12, plb2, prf2, pd2, splb1, splb2, sprf2, spd2 = precompute_conditional(T, ctr, ctr.current_node)

        eng_l = [@view eng_l[m][:, ∂v[2 * j - 1 + m]] for m ∈ 1:2]
        eng_lu = [@view eng_lu[m][:, ∂v[2 * j + 1 + m]] for m ∈ 1:2]
        eng_u = [@view eng_u[m][:, ∂v[2 * j + 3 + m]] for m ∈ 1:2]
        eng_12 = @view eng_12[∂v[2 * j + 6], :]

        le = eng_loc .+ eng_l[1] .+ eng_l[2] .+ eng_lu[1] .+ eng_lu[2] .+ eng_u[1] .+ eng_u[2] .+ eng_12
        ele = exp.(-β .* (le .- minimum(le)))

        @cast LMX5[x, y, v, p1, p2] := LMX[(x, y), (v, p1, p2)]  (p1 ∈ 1:splb1, p2 ∈ 1:splb2, x ∈ 1:1)
        LMX3 =  LMX5[:, :, ∂v[2 * j - 1], ∂v[2 * j + 7], :]
        @cast M4[x, y, p1, p2] := M[x, y, (p1, p2)] (p2 ∈ 1:spd2)
        M2 =  M4[:, :, ∂v[2 * j + 8], :]
        @cast R4[x, y, p1, p2] := R[(x, y), (p1, p2)] (p2 ∈ 1:sprf2, y ∈ 1:1)
        R2 =  R4[:, :, ∂v[2 * j + 9], :]
        LR = dropdims(LMX3[:, :, plb2] ⊠ M2[:, :, pd2] ⊠ R2[:, :, prf2], dims=(1, 2))
        probs = Array(ele .* LR)
    end
    push!(ctr.statistics, ((i, j), ∂v) => error_measure(probs))
    normalize_probability(probs)
end


"""
$(TYPEDSIGNATURES)
"""
function nodes_search_order_Mps(peps::PEPSNetwork{T, S}) where {T <: SquareStar2, S}
    ([(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2], (peps.nrows+1, 1, 1))
end

"""
$(TYPEDSIGNATURES)
"""
function boundary(::Type{T}, ctr::MpsContractor{S}, node::Node) where {T <: SquareStar2, S}
    i, j, k = node  # todo
    if k == 1
        bnd = vcat(
            [
                [((i, m-1, 1), ((i+1, m, 1), (i+1, m, 2)), (i, m-1, 2), ((i+1, m, 1), (i+1, m, 2)),
                (i, m, 1), ((i+1, m-1, 1), (i+1, m-1, 2)), (i, m, 2), ((i+1, m-1, 1), (i+1, m-1, 2))),
                ((i, m, 1), ((i+1, m, 1), (i+1, m, 2)), (i, m, 2), ((i+1, m, 1), (i+1, m, 2)))]
                for m ∈ 1:(j-1)
            ]...,
            ((i, j-1, 1), ((i+1, j, 1), (i+1, j, 2)), (i, j-1, 2), ((i+1, j, 1), (i+1, j, 2))),
            ((i, j-1, 1), (i, j, 1)),
            ((i, j-1, 1), (i, j, 2)),
            ((i, j-1, 2), (i, j, 1)),
            ((i, j-1, 2), (i, j, 2)),
            ((i-1, j-1, 1), (i, j, 1)),
            ((i-1, j-1, 1), (i, j, 2)),
            ((i-1, j-1, 2), (i, j, 1)),
            ((i-1, j-1, 2), (i, j, 2)),
            ((i-1, j, 1), (i, j, 1)),
            ((i-1, j, 1), (i, j, 2)),
            ((i-1, j, 2), (i, j, 1)),
            ((i-1, j, 2), (i, j, 2)),
            [
                [((i-1, m-1, 1), ((i, m, 1), (i, m, 2)), (i-1, m-1, 2), ((i, m, 1), (i, m, 2)),
                (i-1, m, 1), ((i, m-1, 1), (i, m-1, 2)), (i-1, m, 2), ((i, m-1, 1), (i, m-1, 2))),
                ((i-1, m, 1), ((i, m, 1), (i, m, 2)), (i-1, m, 2), ((i, m, 1), (i, m, 2)))]
                for m ∈ (j+1):ctr.peps.ncols
            ]...
        )
    else  # k == 2
        bnd = vcat(
            [
                [((i, m-1, 1), ((i+1, m, 1), (i+1, m, 2)), (i, m-1, 2), ((i+1, m, 1), (i+1, m, 2)),
                (i, m, 1), ((i+1, m-1, 1), (i+1, m-1, 2)), (i, m, 2), ((i+1, m-1, 1), (i+1, m-1, 2))),
                ((i, m, 1), ((i+1, m, 1), (i+1, m, 2)), (i, m, 2), ((i+1, m, 1), (i+1, m, 2)))]
                for m ∈ 1:(j-1)
            ]...,
            ((i, j-1, 1), ((i+1, j, 1), (i+1, j, 2)), (i, j-1, 2), ((i+1, j, 1), (i+1, j, 2))),
            ((i, j-1, 1), (i, j, 2)),
            ((i, j-1, 2), (i, j, 2)),
            ((i-1, j-1, 1), (i, j, 2)),
            ((i-1, j-1, 2), (i, j, 2)),
            ((i-1, j, 1), (i, j, 2)),
            ((i-1, j, 2), (i, j, 2)),
            ((i, j, 1), (i, j, 2)),
            ((i, j, 1), ((i+1, j-1, 1), (i+1, j-1, 2))),
            ((i, j, 1), ((i+1, j, 1), (i+1, j, 2))),
            ((i, j, 1), ((i+1, j+1, 1), (i+1, j+1, 2), (i, j+1, 1), (i, j+1, 2), (i-1, j+1, 1), (i-1, j+1, 2))),
            [
                [((i-1, m-1, 1), ((i, m, 1), (i, m, 2)), (i-1, m-1, 2), ((i, m, 1), (i, m, 2)),
                (i-1, m, 1), ((i, m-1, 1), (i, m-1, 2)), (i-1, m, 2), ((i, m-1, 1), (i, m-1, 2))),
                ((i-1, m, 1), ((i, m, 1), (i, m, 2)), (i-1, m, 2), ((i, m, 1), (i, m, 2)))]
                for m ∈ (j+1):ctr.peps.ncols
            ]...
        )
    end
    bnd
end

"""
$(TYPEDSIGNATURES)
"""
function update_energy(
    ::Type{T}, ctr::MpsContractor{S}, σ::Vector{Int}
) where {T <: SquareStar2, S}
    net = ctr.peps
    i, j, k = ctr.current_node

    en = local_energy(net, (i, j, k))
    for v ∈ ((i, j-1, 1), (i, j-1, 2), (i-1, j, 1), (i-1, j, 2), (i-1, j-1, 1), (i-1, j-1, 2), (i-1, j+1, 1), (i-1, j+1, 2))
        en += bond_energy(net, (i, j, k), v, local_state_for_node(ctr, σ, v))
    end
    if k != 2 return en end
    en += bond_energy(net, (i, j, k), (i, j, 1), local_state_for_node(ctr, σ, (i, j, 1)))  # here k=2
    en
end

# """
# $(TYPEDSIGNATURES)
# """
# function tensor(
#     net::PEPSNetwork{T, Dense}, node::PEPSNode, β::Real, ::Val{:central_d2}
# ) where {T <: AbstractGeometry}
#     i, j = floor(Int, node.i), floor(Int, node.j)
#     T_NW_SE = dense_central_tensor(CentralTensor(net, β, (i, j), (i+1, j+1)))
#     T_NE_SW = dense_central_tensor(CentralTensor(net, β, (i, j+1), (i+1, j)))
#     @cast A[(u, uu), (dd, d)] := T_NW_SE[u, d] * T_NE_SW[uu, dd]
#     A
# end


"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{T, Sparse}, node::PEPSNode, β::Real, ::Val{:central_d2}
) where {T <: AbstractGeometry}
    i, j = floor(Int, node.i), floor(Int, node.j)
    T_NW_SE = CentralTensor(net, β, (i, j), (i+1, j+1))
    T_NE_SW = CentralTensor(net, β, (i, j+1), (i+1, j))
    DiagonalTensor(T_NW_SE, T_NE_SW)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    net::PEPSNetwork{SquareStar2{T}, S}, node::PEPSNode, ::Val{:central_d2}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    s_1 =  CentralTensor_size(net, (i, j), (i+1, j+1))
    s_2 =  CentralTensor_size(net, (i, j+1), (i+1, j))
    (s_1[1] * s_2[1], s_1[2] * s_2[2])
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(  #TODO
    net::PEPSNetwork{SquareStar2{T}, S}, node::PEPSNode, β::Real, ::Val{:sparse_virtual2}
) where {T <: AbstractTensorsLayout, S <: Union{Sparse, Dense}}
    v = Node(node)
    i, j = node.i, floor(Int, node.j)

    p_lb = [projector(net, (i, j, k), ((i+1, j+1, 1), (i+1, j+1, 2))) for k ∈ 1:2]
    p_l = [projector(net, (i, j, k), ((i, j+1, 1), (i, j+1, 2))) for k ∈ 1:2]
    p_lt = [projector(net, (i, j, k), ((i-1, j+1, 1), (i-1, j+1, 2))) for k ∈ 1:2]

    p_rb = [projector(net, (i, j+1, k), ((i+1, j, 1), (i+1, j, 2))) for k ∈ 1:2]
    p_r = [projector(net, (i, j+1, k), ((i, j, 1), (i, j, 2))) for k ∈ 1:2]
    p_rt = [projector(net, (i, j+1, k), ((i-1, j, 1), (i-1, j, 2))) for k ∈ 1:2]

    p_lb[1], p_l[1], p_lt[1] = last(fuse_projectors((p_lb[1], p_l[1], p_lt[1])))
    p_lb[2], p_l[2], p_lt[2] = last(fuse_projectors((p_lb[2], p_l[2], p_lt[2])))

    p_rb[1], p_r[1], p_rt[1] = last(fuse_projectors((p_rb[1], p_r[1], p_rt[1])))
    p_rb[2], p_r[2], p_rt[2] = last(fuse_projectors((p_rb[2], p_r[2], p_rt[2])))

    p_lb = outer_projector(p_lb[1], p_lb[2])
    p_l = outer_projector(p_l[1], p_l[2])
    p_lt = outer_projector(p_lt[1], p_lt[2])

    p_rb = outer_projector(p_rb[1], p_rb[2])
    p_r = outer_projector(p_r[1], p_r[2])
    p_rt = outer_projector(p_rt[1], p_rt[2])

    VirtualTensor(
        net.lp,
        CentralTensor(net, β, (i, j), (i, j+1)),
        (p_lb, p_l, p_lt, p_rb, p_r, p_rt))
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{T, Dense}, node::PEPSNode, β::Real, ::Val{:virtual2}
) where{T <: AbstractGeometry}
    sp = tensor(net, node, β, Val(:sparse_virtual2))
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = (get_projector!(sp.lp, x) for x in sp.projs)
    dense_con = dense_central_tensor(sp.con)

    A = zeros(
        eltype(dense_con),
        length(p_l), maximum.((p_lt, p_rt))..., length(p_r), maximum.((p_lb, p_rb))...
    )
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
         A[l, p_lt[l], p_rt[r], r, p_lb[l], p_rb[r]] = dense_con[p_l[l], p_r[r]]
    end
    @cast B[l, (uu, u), r, (dd, d)] := A[l, uu, u, r, dd, d]
    B
end


"""
$(TYPEDSIGNATURES)
"""
function projectors_site_tensor(
    net::PEPSNetwork{T, S}, vertex::Node
) where {T <: SquareStar2, S}
    i, j = vertex
    plf = outer_projector((projector(net, (i, j, k), ((i+1, j-1, 1), (i+1, j-1, 2), (i, j-1, 1), (i, j-1, 2), (i-1, j-1, 1), (i-1, j-1, 2))) for k ∈ 1:2)...)
    pt = outer_projector((projector(net, (i, j, k), ((i-1, j, 1), (i-1, j, 2))) for k ∈ 1:2)...)
    prf = outer_projector((projector(net, (i, j, k), ((i+1, j+1, 1), (i+1, j+1, 2), (i, j+1, 1), (i, j+1, 2), (i-1, j+1, 1), (i-1, j+1, 2))) for k ∈ 1:2)...)
    pb = outer_projector((projector(net, (i, j, k), ((i+1, j, 1), (i+1, j, 2))) for k ∈ 1:2)...)
    (plf, pt, prf, pb)
end



function Base.size(
    net::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Union{Val{:virtual2}, Val{:sparse_virtual2}}
)
    v = Node(node)
    i, j = node.i, floor(Int, node.j)

    p_lb = [projector(net, (i, j, k), ((i+1, j+1, 1), (i+1, j+1, 2))) for k ∈ 1:2]
    p_l = [projector(net, (i, j, k), ((i, j+1, 1), (i, j+1, 2))) for k ∈ 1:2]
    p_lt = [projector(net, (i, j, k), ((i-1, j+1, 1), (i-1, j+1, 2))) for k ∈ 1:2]

    p_rb = [projector(net, (i, j+1, k), ((i+1, j, 1), (i+1, j, 2))) for k ∈ 1:2]
    p_r = [projector(net, (i, j+1, k), ((i, j, 1), (i, j, 2))) for k ∈ 1:2]
    p_rt = [projector(net, (i, j+1, k), ((i-1, j, 1), (i-1, j, 2))) for k ∈ 1:2]

    p_lb[1], p_l[1], p_lt[1] = last(fuse_projectors((p_lb[1], p_l[1], p_lt[1])))
    p_lb[2], p_l[2], p_lt[2] = last(fuse_projectors((p_lb[2], p_l[2], p_lt[2])))

    p_rb[1], p_r[1], p_rt[1] = last(fuse_projectors((p_rb[1], p_r[1], p_rt[1])))
    p_rb[2], p_r[2], p_rt[2] = last(fuse_projectors((p_rb[2], p_r[2], p_rt[2])))

    p_lb = outer_projector(p_lb[1], p_lb[2])
    p_l = outer_projector(p_l[1], p_l[2])
    p_lt = outer_projector(p_lt[1], p_lt[2])

    p_rb = outer_projector(p_rb[1], p_rb[2])
    p_r = outer_projector(p_r[1], p_r[2])
    p_rt = outer_projector(p_rt[1], p_rt[2])

    (size(p_l, 1), maximum(p_lt) * maximum(p_rt), size(p_r, 1), maximum(p_lb) * maximum(p_rb))
end
