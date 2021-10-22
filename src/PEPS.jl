export 
    PEPSNetwork, 
    node_from_index, 
    conditional_probability

const Node = NTuple{2, Int}

struct GibbsNetwork{T <: AbstractGeometry} <: AbstractGibbsNetwork{Node, Node}
    factor_graph::LabelledGraph{S, Node} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    tensor_map::Dict
    gauges::Dict

    function GibbsNetwork{T}(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation;
    ) where T <: AbstractGeometry

        vmap = vertex_map(transformation, m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(factor_graph, network_graph(T, m, n))
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        tmap = tensor_map(T, nrows, ncols)
        gmap = initialize_gauges(T, nrows, ncols)

        new(factor_graph, vmap, m, n, nrows, ncols, tmap, gmap)
    end
end


# to be removed
function peps_lattice(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end

# to be removed
struct PEPSNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
    factor_graph::LabelledGraph{T, NTuple{2, Int}} where T
    network_graph::LabelledGraph{S, NTuple{2, Int}} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
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

    function PEPSNetwork(
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
        ng = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        _tensors_map = Dict()
        for i ∈ 1:nrows, j ∈ 1:ncols
            push!(_tensors_map, (i, j) => :site)
            push!(_tensors_map, (i, j + 1//2) => :central_h)
            push!(_tensors_map, (i + 1//2, j) => :central_v)
        end
        for i ∈ 1:nrows-1, j ∈ 1:ncols
            push!(_tensors_map, (i + 4//6, j) => :gauge_h)
            push!(_tensors_map, (i + 5//6, j) => :gauge_h)
        end

        _mpo_main = Dict()
        for i ∈ 1:ncols push!(_mpo_main, i => (-1//6, 0, 3//6, 4//6)) end
        for i ∈ 1:ncols - 1 push!(_mpo_main, i + 1//2 => (0,)) end  # consier changing (0,) to 0

        # MPO : Dict(Q => tensor, Q => lista tensorow)
        # gdzie sie da to w 2gim przypadku iteracja po liscie da dispatch do operacji na pojedynczym tensor_size

        _mpo_dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

        _mpo_right = Dict()
        for i ∈ 1:ncols push!(_mpo_right, i => (-3//6, 0)) end
        for i ∈ 1:ncols - 1 push!(_mpo_right, i + 1//2 => (0,)) end  # consier changing (0,) to 0

        _gauges = Dict()

        network = new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim,
                      var_tol, sweeps, _gauges, _tensors_map,
                      _mpo_main, _mpo_dress, _mpo_right
                )
        update_gauges!(network, :id)
        network
    end
end


# function projectors(network::GibbsNetwork{T <: Square}, vertex::NTuple{2, Int}) 
function projectors(network::PEPSNetwork, vertex::NTuple{2, Int})  # wspolne dla siatek dradratowych
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
end


node_index(peps::AbstractGibbsNetwork, node::NTuple{2, Int}) = peps.ncols * (node[1] - 1) + node[2]
_mod_wo_zero(k, m) = k % m == 0 ? m : k % m


node_from_index(peps::AbstractGibbsNetwork, index::Int) =
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))


# function boundary(network::GibbsNetwork{T <: Square}, node::NTuple{2, Int}) 
function boundary(peps::PEPSNetwork, node::NTuple{2, Int})   # ale zwiazane z kolejnoscia szukania przez node_from_index
    i, j = node
    vcat(
        [
            ((i, k), (i+1, k)) for k ∈ 1:j-1
        ]...,
            ((i, j-1), (i, j)),
        [
            ((i-1, k), (i, k)) for k ∈ j:peps.ncols
        ]...
    )
end


function conditional_probability(peps::PEPSNetwork, w::Vector{Int})
    i, j = node_from_index(peps, length(w)+1)   # ale zwiazane z kolejnoscia szukania przez node_from_index
    ∂v = boundary_state(peps, w, (i, j))

    L = left_env(peps, i, ∂v[1:j-1])
    R = right_env(peps, i, ∂v[j+2 : peps.ncols+1])
    A = reduced_site_tensor(peps, (i, j), ∂v[j], ∂v[j+1])

    ψ = dressed_mps(peps, i)
    M = ψ.tensors[j]

    @tensor prob[σ] := L[x] * M[x, d, y] * A[r, d, σ] *
                       R[y, r] order = (x, d, r, y)

    normalize_probability(prob)
end


function bond_energy(
    network::AbstractGibbsNetwork, 
    u::NTuple{2, Int}, 
    v::NTuple{2, Int}, 
    σ::Int
)
    fg_u, fg_v = network.vertex_map(u), network.vertex_map(v)
    if has_edge(network.factor_graph, fg_u, fg_v)
        pu, en, pv = get_prop.(Ref(network.factor_graph), Ref(fg_u), Ref(fg_v), (:pl, :en, :pr))
        energies = en[pu, pv[σ]]
    elseif has_edge(network.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(Ref(network.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr))
        energies = en[pv[σ], pu]
    else
        energies = zeros(length(local_energy(network, u)))
    end
    vec(energies)
end


function update_energy(network::PEPSNetwork, σ::Vector{Int})  #siatka kwadratowa niezaleznie od sposobu zwezania
    i, j = node_from_index(network, length(σ)+1)  # ale zwiazane z kolejnoscia szukania przez node_from_index
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    local_energy(network, (i, j))
end
