export 
    PEPSNetwork, 
    contract_network,
    generate_boundary, 
    node_from_index, 
    drop_physical_index,
    initialize_MPS,
    conditional_probability,
    generate_gauge,
    MPO_connecting,
    MPO_gauge,
    update_gauges!


const DEFAULT_CONTROL_PARAMS = Dict(
    "bond_dim" => typemax(Int),
    "var_tol" => 1E-8,
    "sweeps" => 4.,
    "β" => 1.
)


function peps_lattice(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


@memoize Dict function _right_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int}) 
    M = MPO_connecting(peps, i - 1//2) 
    W = MPO(peps, i)
    ψ = MPS(peps, i+1)
    right_env(ψ, M * W, ∂v)
end


@memoize Dict function _left_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int})
    ψ = MPS(peps, i+1)
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
    gauges::Dict{Tuple{Rational{Int}, Int}, Vector{Real}}


    function PEPSNetwork(
        m::Int,
        n::Int,
        factor_graph,
        transformation::LatticeTransformation;
        β::Real,
        bond_dim::Int=typemax(Int),
        var_tol::Real=1E-8,
        sweeps::Int=4
    )
        vmap = vertex_map(transformation, m, n)
        ng = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end
        gauges = Dict()
        network = new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim, var_tol, sweeps, gauges)
        update_gauges!(network, :id)
        network
    end
end


function projectors(network::PEPSNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
end


function SpinGlassTensors.MPO(::Type{T},
    peps::PEPSNetwork,
    i::Int
) where {T <: Number}
    W = MPO(T, peps.ncols)

    for j ∈ 1:peps.ncols
        A = central_tensor(peps, (i, j))
        Ã = dropdims(sum(A, dims=5), dims=5)
        h = connecting_tensor(peps, (i, j-1), (i, j))
        @tensor B[l, u, r, d] := h[l, l̃] * Ã[l̃, u, r, d]
        W[j] = B
    end
    W
end


@memoize Dict SpinGlassTensors.MPO(
    peps::PEPSNetwork,
    i::Int
) = MPO(Float64, peps, i)


function MPO_connecting(::Type{T},
    peps::PEPSNetwork,
    r::Rational{Int}  # r == n + 1//2
) where {T <: Number}
    W = MPO(T, peps.ncols)
    for j ∈ 1:peps.ncols
        v = connecting_tensor(peps, (floor(Int, r), j), (ceil(Int, r), j))
        @cast A[_, u, _, d] := v[u, d]
        W[j] = A
    end
    W
end


@memoize Dict MPO_connecting(
    peps::PEPSNetwork,
    r::Rational{Int}
) = MPO_connecting(Float64, peps, r)


function update_gauges!(
    network::AbstractGibbsNetwork,
    type::Symbol=:rand
)
    for i ∈ 1:network.nrows - 1, j ∈ 1:network.ncols
        d1, d2 = size(interaction_energy(network, (i, j), (i + 1, j)))
        Y = type == :id ? ones(d1) : rand(d1) .+ 0.1
        push!(network.gauges, (i + 1//6, j) => Y)
        push!(network.gauges, (i + 2//6, j) => 1 ./ Y)
        Z = type == :id ? ones(d2) : rand(d2) .+ 0.1
        push!(network.gauges, (i + 4//6, j) => Z)
        push!(network.gauges, (i + 5//6, j) => 1 ./ Z)
    end
    for j ∈ 1:network.ncols
        push!(network.gauges, (network.nrows + 1//6, j) => ones(1))
        push!(network.gauges, (-1//6, j) => ones(1))
    end
end


function MPO_gauge(::Type{T},
    network::PEPSNetwork,
    r::Rational{Int}
) where {T <: Number}
    W = MPO(T, network.ncols)
    for j ∈ 1:network.ncols
        X = network.gauges[(r, j)]
        @cast A[_, u, _, d] := Diagonal(X)[u, d]
        W[j] = A
    end
    W
end


@memoize Dict MPO_gauge(
network::PEPSNetwork,
r::Rational{Int}
) = MPO_gauge(Float64, network, r)


function compress(
    ψ::AbstractMPS,
    peps::AbstractGibbsNetwork;
)
    if bond_dimension(ψ) < peps.bond_dim return ψ end
    SpinGlassTensors.compress(ψ, peps.bond_dim, peps.var_tol, peps.sweeps)
end


@memoize Dict function SpinGlassTensors.MPS(peps::AbstractGibbsNetwork, i::Int)
    if i > peps.nrows return IdentityMPS() end
    ψ = MPS(peps, i+1)

    #new concept
    #for r ∈ layers ψ *= MPO(peps, r) end
    #compress(ψ, peps)

    Y = MPO_gauge(peps, i + 1 - 5//6)
    W = MPO(peps, i) 
    M = MPO_connecting(peps, i - 3//6)
    X = MPO_gauge(peps, i - 4//6)
    compress((((X * M) * W) * Y) * ψ, peps)
end


contract_network(peps::AbstractGibbsNetwork) = prod(dropindices(MPS(peps, 1)))[]

node_index(peps::AbstractGibbsNetwork, node::NTuple{2, Int}) = peps.ncols * (node[1] - 1) + node[2]

_mod_wo_zero(k, m) = k % m == 0 ? m : k % m


node_from_index(peps::AbstractGibbsNetwork, index::Int) =
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))


function boundary_at_splitting_node(peps::PEPSNetwork, node::NTuple{2, Int})
    i, j = node
    [
        [((i, k), (i+1, k)) for k ∈ 1:j-1]...,
        ((i, j-1), (i, j)),
        [((i-1, k), (i, k)) for k ∈ j:peps.ncols]...
    ]
end


function _normalize_probability(prob::Vector{T}) where {T <: Number}
    # exceptions (negative pdo, etc)
    prob / sum(prob)
end


function conditional_probability(peps::PEPSNetwork, w::Vector{Int})
    i, j = node_from_index(peps, length(w)+1)
    ∂v = boundary_state(peps, w, (i, j))

    L = _left_env(peps, i, ∂v[1:j-1])
    R = _right_env(peps, i, ∂v[j+2:peps.ncols+1])

    h = connecting_tensor(peps, (i, j-1), (i, j))
    v = connecting_tensor(peps, (i-1, j), (i, j))

    l, u = ∂v[j:j+1]

    h̃ = @view h[l, :]
    ṽ = @view v[u, :]
    A = central_tensor(peps, (i, j))

    @tensor B[r, d, σ] := h̃[l] * ṽ[u] * A[l, u, r, d, σ]

    ψ = MPS(peps, i+1)
    M = ψ[j]
    @tensor prob[σ] := L[x] * M[x, d, y] *
                       B[r, d, σ] * R[y, r] order = (x, d, r, y)

    _normalize_probability(prob)
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