export
       site,
       Square,
       tensor_map,
       gauges_list,
       MpoLayers,
       conditional_probability,
       projectors,
       node_from_index,
       index_from_node,
       MPS_contractor_iteration_order,
       boundary,
       update_energy

"Defines Square geometry with a given layout."
struct Square{T <: AbstractTensorsLayout} <: AbstractGeometry end

"Creates Square geometry as a LabelledGraph."
function Square(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end

site(::Type{Dense}) = :site
site(::Type{Sparse}) = :sparse_site

"Assigns type of tensor to a PEPS node coordinates for a given Layout and Sparsity."
function tensor_map(
    ::Type{Square{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{GaugesEnergy, EnergyGauges}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site(S))
        if j < ncols push!(map, PEPSNode(i, j + 1//2) => :central_h) end
        if i < nrows push!(map, PEPSNode(i + 1//2, j) => :central_v) end
    end
    map
end

"Assigns type of tensor to a PEPS node coordinates for a given Layout and Sparsity."
function tensor_map(
    ::Type{Square{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: EngGaugesEng, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site(S))
        if j < ncols push!(map, PEPSNode(i, j + 1//2) => :central_h) end
        if i < nrows
            push!(
                map,
                PEPSNode(i + 1//5, j) => :sqrt_up,
                PEPSNode(i + 4//5, j) => :sqrt_down
            )
         end
    end
    map
end

"Assigns gauges and corresponding information to GaugeInfo structure for a given Layout."
function gauges_list(::Type{Square{T}}, nrows::Int, ncols::Int) where T <: GaugesEnergy
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

"Assigns gauges and corresponding information to GaugeInfo structure for a given Layout."
function gauges_list(::Type{Square{T}}, nrows::Int, ncols::Int) where T <: EnergyGauges
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

"Assigns gauges and corresponding information to GaugeInfo structure for a given Layout."
function gauges_list(::Type{Square{T}}, nrows::Int, ncols::Int) where T <: EngGaugesEng
    [
        GaugeInfo(
            (PEPSNode(i + 2//5, j), PEPSNode(i + 3//5, j)),
            PEPSNode(i + 1//5, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"Defines the MPO layers for the Square geometry with the EnergyGauges layout."
function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EnergyGauges}
    main = Dict{Site, Sites}(i => (-1//6, 0, 3//6, 4//6) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-3//6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (3//6, 4//6) for i ∈ 1:ncols), right)
end

"Defines the MPO layers for the Square geometry with the GaugesEnergy layout."
function MpoLayers(::Type{T}, ncols::Int) where T <: Square{GaugesEnergy}
    main = Dict{Site, Sites}(i => (-4//6, -1//2, 0, 1//6) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-3//6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (1//6,) for i ∈ 1:ncols), right)
end

"Defines the MPO layers for the Square geometry with the EngGaugesEng layout."
function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EngGaugesEng}
    main = Dict{Site, Sites}(i => (-2//5, -1//5, 0, 1//5, 2//5) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end

    right = Dict{Site, Sites}(i => (-4//5, -1//5, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end

    MpoLayers(main, Dict(i => (1//5, 2//5) for i ∈ 1:ncols), right)
end

"Calculates conditional probability for a Square Layout."
function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, state::Vector{Int},
) where {T <: Square, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j = node_from_index(ctr.peps, length(state)+1)
    ∂v = boundary_state(ctr.peps, state, (i, j))

    L = left_env(ctr, i, ∂v[1:j-1], indβ)
    R = right_env(ctr, i, ∂v[(j+2):(ctr.peps.ncols+1)], indβ)
    M = dressed_mps(ctr, i, indβ)[j]

    L ./= maximum(abs.(L))
    R ./= maximum(abs.(R))
    M ./= maximum(abs.(M))

    @tensor LM[y, z] := L[x] * M[x, y, z]
    eng_local = local_energy(ctr.peps, (i, j))

    pl = projector(ctr.peps, (i, j), (i, j-1))
    eng_pl = interaction_energy(ctr.peps, (i, j), (i, j-1))
    eng_left = @view eng_pl[pl[:], ∂v[j]]

    pu = projector(ctr.peps, (i, j), (i-1, j))
    eng_pu = interaction_energy(ctr.peps, (i, j), (i-1, j))
    eng_up = @view eng_pu[pu[:], ∂v[j+1]]

    en = eng_local .+ eng_left .+ eng_up
    en_min = minimum(en)
    loc_exp = exp.(-β .* (en .- en_min))

    pr = projector(ctr.peps, (i, j), (i, j+1))
    pd = projector(ctr.peps, (i, j), (i+1, j))

    bnd_exp = dropdims(sum(LM[pd[:], :] .* R[:, pr[:]]', dims=2), dims=2)
    probs = loc_exp .* bnd_exp
    push!(ctr.statistics, state => error_measure(probs))
    normalize_probability(probs)
end

"Returns rojectors."
function projectors(network::PEPSNetwork{T, S}, vertex::Node) where {T <: Square, S}
    i, j = vertex
    projector.(Ref(network), Ref(vertex), ((i, j-1), (i-1, j), (i, j+1), (i+1, j)))
end

function index_from_node(peps::PEPSNetwork{T, S}, node::Node) where {T <: Square, S}
    peps.ncols * (node[begin] - 1) + node[end]
end

function node_from_index(peps::PEPSNetwork{T, S}, index::Int) where {T <: Square, S}
    ((index - 1) ÷ peps.ncols + 1, mod_wo_zero(index, peps.ncols))
end

function MPS_contractor_iteration_order(peps::PEPSNetwork{T, S}) where {T <: Square, S}
    [(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols]
end

function boundary(peps::PEPSNetwork{T, S}, node::Node) where {T <: Square, S}
    i, j = node
    vcat(
        [((i, k), (i+1, k)) for k ∈ 1:j-1]...,
        ((i, j-1), (i, j)),
        [((i-1, k), (i, k)) for k ∈ j:peps.ncols]...
    )
end

function update_energy(net::PEPSNetwork{T, S}, σ::Vector{Int})  where {T <: Square, S}
    i, j = node_from_index(net, length(σ)+1)
    en = local_energy(net, (i, j))
    for v ∈ ((i, j-1), (i-1, j))
        en += bond_energy(net, (i, j), v, local_state_for_node(net, σ, v))
    end
    en
end
