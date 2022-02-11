export Pegasus

struct Pegasus <: AbstractGeometry end

function Pegasus(m::Int, n::Int)
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

# Geometry: 2 nodes -> 1 TN site. This will work for Chimera.
pegasus_site(::Type{Dense}) = :pegasus_site
pegasus_site(::Type{Sparse}) = :sparse_pegasus_site

function tensor_map(
    ::Type{Pegasus}, ::Type{S}, nrows::Int, ncols::Int
) where S <: AbstractSparsity
    map = Dict{PEPSNode, Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols push!(map, PEPSNode(i, j) => pegasus_site(S)) end
    map
end

function gauges_list(::Type{Pegasus}, nrows::Int, ncols::Int)
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

function MpoLayers(::Type{T}, ncols::Int) where T <: Pegasus
    MpoLayers(
        Dict(i => (-1//3, 0, 1//3) for i ∈ 1:ncols),
        Dict(i => (1//3,) for i ∈ 1:ncols),
        Dict(i => (0,) for i ∈ 1:ncols)
    )
end


function conditional_probability(
    ::Type{T}, ctr::MpsContractor{S}, state::Vector{Int},
) where {T <: Pegasus, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j, k = ctr.current_node
    ∂v = boundary_state(ctr.peps, state, (i, j))

    L = left_env(ctr, i, ∂v[1:j-1], indβ)
    R = right_env(ctr, i, ∂v[(j+2):(ctr.peps.ncols+1)], indβ)
    M = dressed_mps(ctr, i, indβ)[j]

    L ./= maximum(abs.(L))
    R ./= maximum(abs.(R))
    M ./= maximum(abs.(M))

    @tensor LM[y, z] := L[x] * M[x, y, z]

    _, (pl1, pl2) = fuse_projectors([projector(ctr.peps, (i, j-1, 2), (i, j, k)) for k ∈ 1:2])
    _, (pu1, pu2) = fuse_projectors([projector(ctr.peps, (i-1, j, 1), (i, j, k)) for k ∈ 1:2])

    pl = [projector(ctr.peps, (i, j, k), (i, j-1 ,2)) for k ∈ 1:2]
    pu = [projector(ctr.peps, (i, j, k), (i-1, j, 1)) for k ∈ 1:2]

    eu = [interaction_energy(ctr.peps, (i, j, k), (i-1, j, 1)) for k ∈ 1:2]
    el = [interaction_energy(ctr.peps, (i, j, k), (i, j-1, 2)) for k ∈ 1:2]

    eng_left = [el[1][pl[1][:], pl2[∂v[j]]], el[2][pl[2][:], pl1[∂v[j]]]]
    eng_up = [eu[1][pu[1][:], pu1[∂v[j+1]]], eu[2][pu[2][:], pu2[∂v[j+1]]]]

    en21 = interaction_energy(ctr.peps, (i, j, 2), (i, j, 1))
    p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
    p12 = projector(ctr.peps, (i, j, 1), (i, j, 2))

    pr = projector(ctr.peps, (i, j, 2), ((i, j+1, 1), (i, j+1, 2)))
    pd = projector(ctr.peps, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))

    eng_local = [local_energy(ctr.peps, (i, j, k)) for k ∈ 1:2]

    if k == 1
        en = [eng_local[k] .+ eng_left[k] .+ eng_up[k] for k ∈ 1:2]

        ten = reshape(en[2], (:, 1)) .+ en21
        ten_min = minimum(ten)
        ten2 = exp.(-β .* (ten .- ten_min))
        ten3 = zeros(size(ten2, 1), maximum(pr), size(ten2, 2))
        for i ∈ pr ten3[:, i, :] += ten2 end

        RT = R * dropdims(sum(ten3, dims=1), dims=1)
        bnd_exp = dropdims(sum(LM[pd[:], :] .* RT', dims=2), dims=2)
        en_min = minimum(en[1])
        loc_exp = exp.(-β .* (en[1] .- en_min))
    elseif k == 2
        en = eng_local[2] .+ eng_left[2] .+ eng_up[2] .+ en21[p21[:], p12[∂v[end]]]
        en_min = minimum(en)
        loc_exp = exp.(-β .* (en .- en_min))
        bnd_exp = dropdims(sum(LM[pd[:], :] .* R[:, pr[:]]', dims=2), dims=2)
    else
        throw(ArgumentError("Number $k of sub-clusters is incorrect for this $T."))
    end

    probs = loc_exp .* bnd_exp
    push!(ctr.statistics, state => error_measure(probs))
    normalize_probability(probs)
end

function projectors(net::PEPSNetwork{T, S}, vertex::Node) where {T <: Pegasus, S}
    i, j = vertex
    (
        projector(net, (i, j-1, 2), ((i, j, 1), (i, j, 2))),
        projector(net, (i-1, j, 1), ((i, j, 1), (i, j, 2))),
        projector(net, (i, j, 2), ((i, j+1, 1), (i, j+1, 2))),
        projector(net, (i, j, 1), ((i+1, j, 1), (i+1, j, 2)))
    )
end

function index_from_node(peps::PEPSNetwork{T, S}, node::Node) where {T <: Pegasus, S}
    2 * peps.ncols * (node[1] - 1) + 2 * (node[2]-1) + node[3]
end


function nodes_search_order_Mps(peps::PEPSNetwork{T, S}) where {T <: Pegasus, S}
    [(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2]
end

function boundary(peps::PEPSNetwork{T, S}, node::Node) where {T <: Pegasus, S}
    i, j = node
    vcat(
        [((i, k, 1), ((i+1, k, 1), (i+1, k, 2))) for k ∈ 1:j-1]...,
        ((i, (j-1), 2), ((i, j, 1), (i, j, 2))),
        [((i-1, k, 1), ((i, k, 1), (i, k, 2))) for k ∈ j:peps.ncols]...,
        ((i, j, 1), (i, j, 2))
    )
end

function update_energy(
    ::Type{T}, ctr::MpsContractor{S}, σ::Vector{Int}) where {T <: Pegasus, S}
    net = ctr.peps
    i, j, k = ctr.current_node
    en = local_energy(net, (i, j, k))
    for v ∈ ((i, j-1, 1), (i-1, j, 2), (i, j-1, 2), (i-1, j, 1))
        en += bond_energy(net, (i, j, k), v, local_state_for_node(net, σ, v))
    end
    if k != 2 return en end
    en += bond_energy(net, (i, j, k), (i, j, 1), local_state_for_node(net, σ, (i, j, 1)))
    en
end

# cluster-cluster energies atttached from left and top
function tensor(
    network::PEPSNetwork{Pegasus, T}, node::PEPSNode, β::Real, ::Val{:pegasus_site}
) where T <: AbstractSparsity
    i, j = node.i, node.j

    en1 = local_energy(network, (i, j, 1))
    en2 = local_energy(network, (i, j, 2))
    en12 = interaction_energy(network, (i, j, 1), (i, j, 2))
    eloc = zeros(length(en1), length(en2))
    p1 = projector(network, (i, j, 1), (i, j, 2))
    p2 = projector(network, (i, j, 2), (i, j, 1))

    for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
        eloc[s2, s1] = en1[s1] + en2[s2] + en12[p1[s1], p2[s2]]
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

    e1u = @view e1u[:, pu1]
    e2u = @view e2u[:, pu2]
    e1l = @view e1l[:, pl1]
    e2l = @view e2l[:, pl2]

    le1u = exp.(-β .* (e1u .- minimum(e1u)))
    le2u = exp.(-β .* (e2u .- minimum(e2u)))
    le1l = exp.(-β .* (e1l .- minimum(e1l)))
    le2l = exp.(-β .* (e2l .- minimum(e2l)))

    A = zeros(maximum.((pl, pu, pr, pd)))
    for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
        ll = reshape(le1l[p1l[s1], :], :, 1) .* reshape(le2l[p2l[s2], :], :, 1)
        lu = reshape(le1u[p1u[s1], :], 1, :) .* reshape(le2u[p2u[s2], :], 1, :)
        A[:, :, pr[s2], pd[s1]] += loc_exp[s2, s1] .* (ll .* lu)
    end
    A
end

# function tensor(
#     network::PEPSNetwork{Pegasus, T}, node::PEPSNode, β::Real, ::Val{:sparse_pegasus_site}
# ) where T <: AbstractSparsity
#     ## TO BE ADDED
# end

function Base.size(
    network::PEPSNetwork{Pegasus, T}, node::PEPSNode, ::Val{:pegasus_site}
) where T <: AbstractSparsity
    maximum.(projectors(network, Node(node)))
end

function Base.size(
    network::PEPSNetwork{Pegasus, T}, node::PEPSNode, ::Val{:sparse_pegasus_site}
) where T <: AbstractSparsity
    maximum.(projectors(network, Node(node)))
end
