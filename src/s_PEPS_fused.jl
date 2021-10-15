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
    #
    β::Real
    #
    bond_dim::Int
    var_tol::Real
    sweeps::Int
    #
    gauges
    tensors_map
    #
    mpo_main::Dict
    mpo_dress::Dict
    mpo_right::Dict
    #
    function FusedNetwork(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation;
        β::Real,
        bond_dim::Int=typemax(Int),
        var_tol::Real=1E-8,
        sweeps::Int=4,
    )
    vmap = vertex_map(transformation, m, n)
    ng = cross_lattice(m, n)
    nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
    
    if !is_compatible(factor_graph, ng)
        throw(ArgumentError("Factor graph not compatible with given network."))
    end

    _gauges = Dict()
    _tensors_map = Dict()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(_tensors_map, (i, j) => :site)
        push!(_tensors_map, (i, j - 1//2) => :virtual)
        push!(_tensors_map, (i + 1//2, j) => :central_v)
    end
    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(_tensors_map, (i + 1//2, j + 1//2) => :central_d)
    end
    for i ∈ 1 : nrows - 1, j ∈ 1//2 : 1//2 : ncols
        jj = denominator(j) == 1 ? numerator(j) : j
        push!(_tensors_map, (i + 4//6, jj) => :gauge_h)
        push!(_tensors_map, (i + 5//6, jj) => :gauge_h)
    end

    _mpo_main = Dict()
    _mpo_right = Dict()
    _mpo_dress = Dict()
    for i ∈ 1//2 :1//2 : ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(_mpo_main, ii => (-1//6, 0, 3//6, 4//6))
        push!(_mpo_dress, ii => (3//6, 4//6))
        push!(_mpo_right, ii => (-3//6, 0))
    end

    network = new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim,
                  var_tol, sweeps, _gauges, _tensors_map,
                  _mpo_main, _mpo_dress, _mpo_right
            )
    update_gauges!(network, :id)
    network
    end
end


function projectors(network::FusedNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    neighbours = (
                    ((i+1, j-1), (i, j-1), (i-1, j-1)), 
                    (i-1, j),
                    ((i+1, j+1), (i, j+1), (i-1, j+1)),
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
        ((i, j-1), (i+1, j)),
        ((i, j-1), (i, j)),
        ((i-1, j-1), (i, j)),
        ((i-1, j), (i, j)),
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
    MX, M = ψ[j-1//2], ψ[j]

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