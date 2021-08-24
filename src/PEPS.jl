export 
    PEPSNetwork, 
    generate_boundary, 
    node_from_index, 
    conditional_probability


function peps_lattice(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


@memoize Dict function _right_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int}) 
    W = prod(mpo.(Ref(peps), i .+ reverse(peps.layers_right_env)))
    ψ = mps(peps, i, :dressed)
    right_env(ψ, W, ∂v)
end


@memoize Dict function _left_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int})
    ψ = mps(peps, i, :dressed)
    left_env(ψ, ∂v)
end


struct PEPSNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
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
    tensor_types
    gauges
    tensor_spiecies
    columns_MPO::NTuple{N, Union{Rational{Int}, Int}} where N
    layers_MPS::NTuple{M, Union{Rational{Int}, Int}} where M
    layers_left_env::NTuple{K, Union{Rational{Int}, Int}} where K
    layers_right_env::NTuple{L, Union{Rational{Int}, Int}} where L

    function PEPSNetwork(
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
        # with gauges but differently
        layers_MPS=(4//6, 3//6, 0, -1//6),  # from bottom to top
        layers_left_env=(4//6, 3//6),
        layers_right_env=(0, -3//6)
    )
        vmap = vertex_map(transformation, m, n)
        ng = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
        
        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        _types = (:site, :central_h, :central_v, :gauge_h)
        _gauges = Dict()
        _tensor_spiecies = Dict()
        
        network = new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim,
                      var_tol, sweeps, _types, _gauges, _tensor_spiecies,
                      columns_MPO, layers_MPS, layers_left_env, layers_right_env
                )
        update_gauges!(network, :id)
        tensor_species_map!(network, network.tensor_types)
        network
    end
end


function projectors(network::PEPSNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
end


node_index(peps::AbstractGibbsNetwork, node::NTuple{2, Int}) = peps.ncols * (node[1] - 1) + node[2]
_mod_wo_zero(k, m) = k % m == 0 ? m : k % m


node_from_index(peps::AbstractGibbsNetwork, index::Int) =
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))


function boundary(peps::PEPSNetwork, node::NTuple{2, Int})
    i, j = node
    x = (-4, -2)
    vcat(
        [
            [(x, x), ((i, k), (i+1, k))] for k ∈ 1:j-1
        ]...,
            (x, x), ((i, j-1), (i, j)),
        [
            [(x, x), ((i-1, k), (i, k))] for k ∈ j:peps.ncols
        ]...
    )
end


function conditional_probability(peps::PEPSNetwork, w::Vector{Int})
    i, j = node_from_index(peps, length(w)+1)
    ∂v = boundary_state(peps, w, (i, j))

    L = _left_env(peps, i, ∂v[1:2*j-1])
    R = _right_env(peps, i, ∂v[2*j+3 : 2*peps.ncols+2])
    A = reduced_site_tensor(peps, (i, j), ∂v[2*j], ∂v[2*j+2])

    ψ = mps(peps, i, :dressed)
    M = ψ[2 * j]

    @tensor prob[σ] := L[x] * M[x, d, y] * A[r, d, σ] *
                       R[y, r] order = (x, d, r, y)

    normalize_probability(peps, prob)
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
        energies = (pu * (en * pv[:, σ:σ]))'
    elseif has_edge(network.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(Ref(network.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr))
        energies = (pv[σ:σ, :] * en) * pu
    else
        energies = zeros(length(local_energy(network, u)))
    end
    vec(energies)
end


function update_energy(network::PEPSNetwork, σ::Vector{Int})
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    local_energy(network, (i, j))
end
