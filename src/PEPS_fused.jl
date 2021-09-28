export FusedNetwork, boundary,
    conditional_probability,
    update_energy,
    projectors


function cross_lattice(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    lg = LabelledGraph(labels, grid((m, n)))
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end


struct FusedNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
    factor_graph::LabelledGraph{T, NTuple{2, Int}} where T
    network_graph::LabelledGraph{S, NTuple{2, Int}} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    β::Real
    bond_dim::Int
    var_tol::Real
    sweeps::Int
    gauges
    tensor_spiecies
    columns_MPO::NTuple{N, Union{Rational{Int}, Int}} where N
    layers_MPS::NTuple{M, Union{Rational{Int}, Int}} where M
    layers_left_env::NTuple{K, Union{Rational{Int}, Int}} where K
    layers_right_env::NTuple{L, Union{Rational{Int}, Int}} where L

    function FusedNetwork(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation;
        β::Real,
        bond_dim::Int=typemax(Int),
        var_tol::Real=1E-8,
        sweeps::Int=4,
        columns_MPO = (-1//2, 0),  # from left to right
        # with gauges
        # layers_MPS=(1//6, 0, -3//6, -4//6),  # from bottom to top
        # layers_left_env=(1//6,),
        # layers_right_env=(0, -3//6)
        # with gauges
         layers_MPS=(4//6, 3//6, 0, -1//6),  # from bottom to top
         layers_left_env=(4//6, 3//6),
         layers_right_env=(0, -3//6)
        # without gauges
        # layers_MPS=(0, -3//6),  # from bottom to top
        # layers_left_env=(),
        # layers_right_env=(0, -3//6)
        # no gauges
        # layers_MPS=(3//6, 0),  # from bottom to top
        # layers_left_env=(3//6,),
        # layers_right_env=(0, -3//6)
    )
    vmap = vertex_map(transformation, m, n)
    ng = cross_lattice(m, n)
    nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)

    if !is_compatible(factor_graph, ng)
        throw(ArgumentError("Factor graph not compatible with given network."))
    end

    _types = (:site, :central_h, :central_v, :virtual, :central_d, :gauge_h)
    _gauges = Dict()

    _tensor_spiecies = Dict()

    network = new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim,
                  var_tol, sweeps, _gauges, _tensor_spiecies,
                  columns_MPO, layers_MPS, layers_left_env, layers_right_env
            )

    for type ∈ _types
        push!(network.tensor_spiecies, tensor_assignment(network, type)...)
    end
    
    update_gauges!(network, :id)
    network
    end
end


function projectors(network::FusedNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    neighbours = (
                    (
                        (i+1, j-1), (i, j-1), (i-1, j-1)
                    ),
                    (i-1, j),
                    (
                        (i+1, j+1), (i, j+1), (i-1, j+1)
                    ),
                    (i+1, j)
                )
    projector.(Ref(network), Ref(vertex), neighbours)
end


function boundary(peps::FusedNetwork, node::NTuple{2, Int})
    i, j = node
    vcat(
        [
            [((i, k-1), (i+1, k), (i, k), (i+1, k-1)), ((i, k), (i+1, k))] for k ∈ 1:j-1
        ]...,
        (
            (i, j-1), (i+1, j)
        ),
        (
            (i, j-1), (i, j)
        ),
        (
            (i-1, j-1), (i, j)
        ),
        (
            (i-1, j), (i, j)
        ),
        [
            [((i-1, k-1), (i, k), (i-1, k), (i, k-1)), ((i-1, k), (i, k))] for k ∈ j+1:peps.ncols
        ]...
    )
end


function conditional_probability(peps::FusedNetwork, w::Vector{Int})
    i, j = node_from_index(peps, length(w)+1)
    ∂v = boundary_state(peps, w, (i, j))

    L = left_env(peps, i, ∂v[1:2*j-2])
    R = right_env(peps, i, ∂v[2*j+3 : 2*peps.ncols+2])
    A = reduced_site_tensor(peps, (i, j), ∂v[2*j-1], ∂v[2*j], ∂v[2*j+1], ∂v[2*j+2])

    ψ = dressed_mps(peps, i)
    MX, M = ψ[2*j-1], ψ[2*j]

    @tensor prob[σ] := L[x] * MX[x, m, y] * M[y, l, z] * R[z, k] *
                        A[k, l, m, σ] order = (x, y, z, k, l, m)

    normalize_probability(prob)
end


function update_energy(network::FusedNetwork, σ::Vector{Int})
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    bond_energy(network, (i, j), (i-1, j-1), local_state_for_node(network, σ, (i-1, j-1))) +
    bond_energy(network, (i, j), (i-1, j+1), local_state_for_node(network, σ, (i-1, j+1))) +
    local_energy(network, (i, j))
end
