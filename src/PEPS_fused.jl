export 
    FusedNetwork, 
    boundary_at_splitting_node,
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
    gauges::Dict{Tuple{Rational{Int}, Rational{Int}}, Vector{Float64}} # Real ?
    tensor_spiecies::Dict{Tuple{Rational{Int}, Rational{Int}}, Symbol}
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
        # layers_MPS=(0, -3//6),  # from bottom to top
        # layers_left_env=(),
        # layers_right_env=(0, -3//6)
        layers_MPS=(3//6, 0),  # from bottom to top
        layers_left_env=(3//6,),
        layers_right_env=(0, -3//6)
    )
    vmap = vertex_map(transformation, m, n)
    ng = cross_lattice(m, n)
    nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
    
    if !is_compatible(factor_graph, ng)
        throw(ArgumentError("Factor graph not compatible with given network."))
    end
    network = new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim,
                  var_tol, sweeps, Dict(), Dict(),
                  columns_MPO, layers_MPS, layers_left_env, layers_right_env
            )
    update_gauges!(network, :id)
    tensor_species_map!(network)
    network
    end
end


function tensor_species_map!(network::FusedNetwork)
    for i ∈ 1:network.nrows, j ∈ 1:network.ncols
        push!(network.tensor_spiecies, (i, j) => :site)
    end
    for i ∈ 1:network.nrows, j ∈ 1:network.ncols - 1
        push!(network.tensor_spiecies, (i, j + 1//2) => :virtual)
    end
    for i ∈ 1:network.nrows-1, j ∈ 1:network.ncols
        push!(network.tensor_spiecies, (i + 1//2, j) => :central_v)
    end
    for i ∈ 1:network.nrows-1, j ∈ 1:network.ncols-1
        push!(network.tensor_spiecies, (i + 1//2, j + 1//2) => :central_d)
    end
    for i ∈ 1:network.nrows-1, r ∈ 1:1//2:network.ncols
        push!(network.tensor_spiecies,
            (i + 1//6, r) => :gauge_h, 
            (i + 2//6, r) => :gauge_h,
            (i + 4//6, r) => :gauge_h, 
            (i + 5//6, r) => :gauge_h,
        )
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


function boundary_at_splitting_node(peps::FusedNetwork, node::NTuple{2, Int})
    i, j = node
    vcat(
        [
            [((i, k-1), (i+1, k), (i, k), (i+1, k-1)), ((i, k), (i+1, k))] for k ∈ 1:j-1
        ]...,
        (
            (i, j-1), ((i+1, j), (i, j), (i-1, j))
        ), 
        [
            [((i-1, k-1), (i, k), (i-1, k), (i, k-1)), ((i-1, k), (i, k))] for k ∈ j:peps.ncols
        ]...
    )
end

# to be simplified
function conditional_probability(peps::FusedNetwork, w::Vector{Int})
    i, j = node_from_index(peps, length(w)+1)
    ∂v = boundary_state(peps, w, (i, j))

    L = _left_env(peps, i, ∂v[1:2*j-2])
    R = _right_env(peps, i, ∂v[2*j+2 : 2*peps.ncols+1])

    A = tensor_temp(peps, (i, j))
    X = tensor(peps, (i, j-1//2))
    Xdiag = tensor(peps, (i - 1//2, j-1//2))

    l, d, u = ∂v[2*j-1:2*j+1]

    X1 = @view Xdiag[1, d, 1, :]
    X2 = @view X[l, :, :, :]
    @tensor Xt[r, d] := X2[x, r, d] * X1[x]

    ev = connecting_tensor(peps, (i-1, j), (i, j)) 
    vt = ev[u, :]
    @tensor Ã[l, r, d, σ] := A[l, x, r, d, σ] * vt[x]

    ψ = MPS(peps, i, :dressed)
    MX, M = ψ[2*j-1], ψ[2*j]

    @tensor prob[σ] := L[x] * Xt[k, y] * MX[x, y, z] * M[z, l, m] *
                       Ã[k, n, l, σ] * R[m, n] order = (x, y, z, k, l, m, n)

    _normalize_probability(prob)
end


function update_energy(network::FusedNetwork, σ::Vector{Int})
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    bond_energy(network, (i, j), (i-1, j-1), local_state_for_node(network, σ, (i-1, j-1))) +
    bond_energy(network, (i, j), (i-1, j+1), local_state_for_node(network, σ, (i-1, j+1))) +
    local_energy(network, (i, j))
end