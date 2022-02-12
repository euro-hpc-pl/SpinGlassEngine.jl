export SquareStar

struct SquareStar{T <: AbstractTensorsLayout} <: AbstractGeometry end

function SquareStar(m::Int, n::Int)
    lg = Square(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end

Virtual(::Type{Dense}) = :virtual
Virtual(::Type{Sparse}) = :sparse_virtual

function tensor_map(
    ::Type{SquareStar{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{EnergyGauges, GaugesEnergy}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site(S),
            PEPSNode(i, j - 1//2) => Virtual(S),
            PEPSNode(i + 1//2, j) => :central_v
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(map, PEPSNode(i + 1//2, j + 1//2) => :central_d)
    end
    map
end

function tensor_map(
    ::Type{SquareStar{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: EngGaugesEng, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site(S),
            PEPSNode(i, j - 1//2) => Virtual(S),
            PEPSNode(i + 1//5, j) => :sqrt_up,
            PEPSNode(i + 4//5, j) => :sqrt_down
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(
            map,
            PEPSNode(i + 1//5, j + 1//2) => :sqrt_up_d,
            PEPSNode(i + 4//5, j + 1//2) => :sqrt_down_d
        )
    end
    map
end

function gauges_list(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T <: GaugesEnergy
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

function gauges_list(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T <: EnergyGauges
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

function gauges_list(::Type{SquareStar{T}}, nrows::Int, ncols::Int) where T <: EngGaugesEng
    [
        GaugeInfo(
            (PEPSNode(i + 2//5, j), PEPSNode(i + 3//5, j)),
            PEPSNode(i + 1//5, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

"Defines the MPO layers for the SquareStar geometry with the EnergyGauges layout."
function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EnergyGauges}
    MpoLayers(
        Dict(site(i) => (-1//6, 0, 3//6, 4//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (3//6, 4//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end

"Defines the MPO layers for the SquareStar geometry with the GaugesEnergy layout."
function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{GaugesEnergy}
    MpoLayers(
        Dict(site(i) => (-4//6, -1//2, 0, 1//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1//6,) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end

"Defines the MPO layers for the SquareStar geometry with the EngGaugesEng layout."
function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EngGaugesEng}
    MpoLayers(
        Dict(site(i) => (-2//5, -1//5, 0, 1//5, 2//5) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1//5, 2//5) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-4//5, -1//5, 0) for i ∈ 1//2:1//2:ncols)
    )
end

# TODO: rewrite this using brodcasting if possible
function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, state::Vector{Int},
) where {T <: SquareStar, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j = ctr.current_node
    ∂v = boundary_state(ctr, state, (i, j))

    L = left_env(ctr, i, ∂v[1:2*j-2], indβ)
    R = right_env(ctr, i, ∂v[(2*j+3):(2*ctr.peps.ncols+2)], indβ)
    ψ = dressed_mps(ctr, i, indβ)
    MX, M = ψ[j-1//2], ψ[j]
    @tensor LMX[y, z] := L[x] * MX[x, y, z]

    eng_local = local_energy(ctr.peps, (i, j))
    pl = projector(ctr.peps, (i, j), (i, j-1))
    eng_pl = interaction_energy(ctr.peps, (i, j), (i, j-1))
    eng_left = @view eng_pl[pl[:], ∂v[2*j]]

    pu = projector(ctr.peps, (i, j), (i-1, j))
    eng_pu = interaction_energy(ctr.peps, (i, j), (i-1, j))
    eng_up = @view eng_pu[pu[:], ∂v[2*j+2]]

    pd = projector(ctr.peps, (i, j), (i-1, j-1))
    eng_pd = interaction_energy(ctr.peps, (i, j), (i-1, j-1))
    eng_diag = @view eng_pd[pd[:], ∂v[2*j+1]]

    en = eng_local .+ eng_left .+ eng_diag .+ eng_up
    en_min = minimum(en)
    loc_exp = exp.(-β .* (en .- en_min))

    p_lb = projector(ctr.peps, (i, j-1), (i+1, j))
    p_rb = projector(ctr.peps, (i, j), (i+1, j-1))
    pr = projector(ctr.peps, (i, j), ((i+1, j+1), (i, j+1), (i-1, j+1)))
    pd = projector(ctr.peps, (i, j), (i+1, j))

    @cast lmx2[b, c, d] := LMX[(b, c), d] (b ∈ 1:maximum(p_lb), c ∈ 1:maximum(p_rb))

    for σ ∈ 1:length(loc_exp)
        lmx = @view lmx2[∂v[2*j-1], p_rb[σ], :]
        m = @view M[:, pd[σ], :]
        r = @view R[:, pr[σ]]
        loc_exp[σ] *= (lmx' * m * r)[]
    end
    push!(ctr.statistics, state => error_measure(loc_exp))
    normalize_probability(loc_exp)
end

function projectors(network::PEPSNetwork{T, S}, vertex::Node) where {T <: SquareStar, S}
    i, j = vertex
    nbrs = (
        ((i+1, j-1), (i, j-1), (i-1, j-1)),
        (i-1, j),
        ((i+1, j+1), (i, j+1), (i-1, j+1)),
        (i+1, j)
    )
    projector.(Ref(network), Ref(vertex), nbrs)
end

function nodes_search_order_Mps(peps::PEPSNetwork{T, S}) where {T <: SquareStar, S}
    [(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols]
end

function boundary(peps::PEPSNetwork{T, S}, node::Node) where {T <: SquareStar, S}
    i, j = node
    vcat(
        [
            [((i, k-1), (i+1, k), (i, k), (i+1, k-1)), ((i, k), (i+1, k))]
            for k ∈ 1:(j-1)
        ]...,
        ((i, j-1), (i+1, j)),
        ((i, j-1), (i, j)),
        ((i-1, j-1), (i, j)),
        ((i-1, j), (i, j)),
        [
            [((i-1, k-1), (i, k), (i-1, k), (i, k-1)), ((i-1, k), (i, k))]
            for k ∈ (j+1):peps.ncols
        ]...
    )
end

function update_energy(
    ::Type{T}, ctr::MpsContractor{S}, σ::Vector{Int}) where {T <: SquareStar, S}
    net = ctr.peps
    i, j = ctr.current_node
    en = local_energy(net, (i, j))
    for v ∈ ((i, j-1), (i-1, j), (i-1, j-1), (i-1, j+1))
        en += bond_energy(net, (i, j), v, local_state_for_node(ctr, σ, v))
    end
    en
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, S}, node::PEPSNode, β::Real, ::Val{:central_d}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1), β)
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j), β)
    @cast A[(u, ũ), (d, d̃)] := NW[u, d] * NE[ũ, d̃]
    A
end

function Base.size(
    network::PEPSNetwork{SquareStar{T}, S}, node::PEPSNode, ::Val{:central_d}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (u * ũ, d * d̃)
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, Dense}, node::PEPSNode, β::Real, ::Val{:virtual}
) where T <: AbstractTensorsLayout
    i, j = node.i, floor(Int, node.j)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    v = Node(node)
    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v), β)

    A = zeros(
        length(p_l), maximum(p_rt), maximum(p_lt),
        length(p_r), maximum(p_lb), maximum(p_rb)
    )

    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        A[l, p_rt[r], p_lt[l], r, p_lb[l], p_rb[r]] = h[p_l[l], p_r[r]]
    end
    @cast AA[l, (ũ, u), r, (d̃, d)] := A[l, ũ, u, r, d̃, d]
    AA
end

function tensor(
    net::PEPSNetwork{SquareStar{T}, Sparse}, node::PEPSNode, β::Real, ::Val{:sparse_virtual}
) where T <: AbstractTensorsLayout
    i, j = node.i, floor(Int, node.j)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(net), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(net), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    v = Node(node)
    h = connecting_tensor(net, floor.(Int, v), ceil.(Int, v), β)

    SparseVirtualTensor(h, (vec(p_lb), vec(p_l), vec(p_lt), vec(p_rb), vec(p_r), vec(p_rt)))
end

function tensor(
    network::PEPSNetwork{SquareStar{T}, Dense}, node::PEPSNode, ::Val{:virtual}
) where T <: AbstractTensorsLayout
    i, s = v
    j = floor(Int, s)

    left_nbrs = ((i+1, j+1), (i, j+1), (i-1, j+1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i+1, j), (i, j), (i-1, j))
    prr = projector.(Ref(network), Ref((i, j+1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    (length(p_l), maximum(p_lt) * maximum(p_rt), length(p_r), maximum(p_rb) * maximum(p_lb))
end
