export PEPSNetwork, contract_network
export generate_boundary, peps_indices

const DEFAULT_CONTROL_PARAMS = Dict(
    "bond_dim" => typemax(Int),
    "var_tol" => 1E-8,
    "sweeps" => 4.,
    "β" => 1.
)

# struct PEPSNetwork <: AbstractGibbsNetwork
#     size::NTuple{2, Int}
#     map::Dict
#     fg::MetaDiGraph
#     nbrs::Dict
#     origin::Symbol
#     i_max::Int
#     j_max::Int
#     β::Number # TODO: get rid of this
#     args::Dict{String, Number}

#     function PEPSNetwork(
#         m::Int,
#         n::Int,
#         fg::MetaDiGraph,
#         β::Number,
#         origin::Symbol=:NW,
#         args_override::Dict{String, T}=Dict{String, Number}()  # TODO: change String to Symbol
#     ) where T <: Number
#         map, i_max, j_max = peps_indices(m, n, origin)

#         # v => (l, u, r, d)
#         nbrs = Dict(
#             map[i, j] => (map[i, j-1], map[i-1, j], map[i, j+1], map[i+1, j])
#             for i ∈ 1:i_max, j ∈ 1:j_max
#         )

#         args = merge(DEFAULT_CONTROL_PARAMS, args_override)
#         pn = new((m, n), map, fg, nbrs, origin, i_max, j_max, β, args)
#     end
# end

function peps_lattice(m, n)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


struct PEPSNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
    factor_graph::LabelledGraph{T, NTuple{2, Int}} where T
    network_graph::LabelledGraph{S, NTuple{2, Int}} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int

    function PEPSNetwork(m::Int, n::Int, factor_graph, transformation::LatticeTransformation)
        vmap = vertex_map(transformation, m, n)
        ng = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
        if !is_compatible(factor_graph, ng, vmap)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end
        new(factor_graph, ng, vmap, m, n, nrows, ncols)
    end
end




function projectors(network::PEPSNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
end


# @memoize function generate_tensor(network::PEPSNetwork, v::Int)
#     # TODO: does this require full network, or can we pass only fg?
#     loc_exp = exp.(-network.β .* get_prop(network.fg, v, :loc_en))

#     dim = zeros(Int, length(network.nbrs[v]))
#     @cast A[_, i] := loc_exp[i]

#     # for proj ∈ _get_projectors(network, v)
#     #    @cast A[(c, γ), σ] |= A[c, σ] * proj[σ, γ]
#     #    dim[j] = size(pv, 2)
#     for (j, w) ∈ enumerate(network.nbrs[v])
#         pv = _get_projector(network.fg, v, w)
#         @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
#         dim[j] = size(pv, 2)
#     end
#     reshape(A, dim..., :)
# end

# @memoize function generate_tensor(network::PEPSNetwork, v::Int, w::Int)
#     fg = network.fg
#     if has_edge(fg, w, v)
#         en = get_prop(fg, w, v, :en)'
#     elseif has_edge(fg, v, w)
#         en = get_prop(fg, v, w, :en)
#     else
#         en = zeros(1, 1)
#     end
#     exp.(-network.β .* (en .- minimum(en)))
# end

function peps_tensor(::Type{T}, peps::PEPSNetwork, i::Int, j::Int, β::Real) where {T <: Number}
    # generate tensors from projectors
    A = build_tensor(peps, (i, j), β)

    # include energy
    h = build_tensor(peps, (i, j-1), (i, j), β)
    v = build_tensor(peps, (i-1, j), (i, j), β)
    @tensor B[l, u, r, d, σ] := h[l, l̃] * v[u, ũ] * A[l̃, ũ, r, d, σ]
    B
end

peps_tensor(peps::PEPSNetwork, i::Int, j::Int, β::Real) = peps_tensor(Float64, peps, i, j, β)

function SpinGlassTensors.PEPSRow(::Type{T}, peps::PEPSNetwork, i::Int, β::Real) where {T <: Number}
    ψ = PEPSRow(T, peps.ncols)
    for j ∈ 1:peps.ncols
        ψ[j] = peps_tensor(T, peps, i, j, β)
    end
    ψ
end
SpinGlassTensors.PEPSRow(peps::PEPSNetwork, i::Int, β::Real) = PEPSRow(Float64, peps, i, β)

function SpinGlassTensors.MPO(::Type{T},
    peps::PEPSNetwork,
    i::Int,
    β::Real,
    config::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) where {T <: Number}

    W = MPO(T, peps.ncols)
    R = PEPSRow(T, peps, i, β)

    for (j, A) ∈ enumerate(R)
        v = get(config, peps.vertex_map((i, j)), nothing)
        if v !== nothing
            @cast B[l, u, r, d] |= A[l, u, r, d, $(v)]
        else
            @reduce B[l, u, r, d] |= sum(σ) A[l, u, r, d, σ]
        end
        W[j] = B
    end
    W
end

SpinGlassTensors.MPO(peps::PEPSNetwork,
    i::Int,
    β::Real,
    config::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) = MPO(Float64, peps, i, β, config)

function compress(
    ψ::AbstractMPS,
    peps::PEPSNetwork;
    bond_dim=typemax(Int),
    var_tol=1E-8,
    sweeps=4
)
    if bond_dimension(ψ) < bond_dim return ψ end
    compress(ψ, bond_dim, var_tol, sweeps)
end


@memoize function SpinGlassTensors.MPS(
    peps::PEPSNetwork,
    i::Int,
    β::Real,
    cfg::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}();
    bond_dim=typemax(Int),
    var_tol=1E-8,
    sweeps=4
)
    if i > peps.nrows return IdentityMPS() end
    W = MPO(peps, i, β, cfg)
    ψ = MPS(peps, i+1, β, cfg)
    compress(W * ψ, peps, bond_dim=bond_dim, var_tol=var_tol, sweeps=sweeps)
end

function contract_network(
    peps::PEPSNetwork,
    β::Real,
    config::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}(),
)
    ψ = MPS(peps, 1, β, config)
    prod(dropindices(ψ))[]
end


@inline function get_coordinates(peps::PEPSNetwork, k::Int)
    ceil(Int, k / peps.ncols), (k - 1) % peps.ncols + 1
end

function generate_boundary(network::PEPSNetwork, v::NTuple{2, Int}, w::NTuple{2, Int}, state::Int)
    if v ∉ vertices(network.network_graph) return 1 end
    loc_dim = length(local_energy(network, v))
    pv = projector(network, v, w)
    findfirst(x -> x > 0, pv[state, :])
end

function generate_boundary(peps::PEPSNetwork, v::Vector{Int}, w::NTuple{2, Int})
    i, j = w
    ∂v = zeros(Int, peps.ncols + 1)

    # on the left below
    for k ∈ 1:j-1
        ∂v[k] = generate_boundary(peps, (i, k), (i+1, k), _get_local_state(peps, v, (i, k)))
    end

    # on the left at the current row
    ∂v[j] = generate_boundary(peps, (i, j-1), (i, j), _get_local_state(peps, v, (i, j-1)))

    # on the right above
    for k ∈ j:peps.ncols
        ∂v[k+1] = generate_boundary(peps, (i-1, k), (i, k), _get_local_state(peps, v, (i-1, k)))
    end
    ∂v
end

function _get_local_state(peps::PEPSNetwork, σ::Vector{Int}, w::NTuple{2, Int})
    k = w[2] + peps.ncols * (w[1] - 1)
    0 < k <= length(σ) ? σ[k] : 1
end

function _normalize_probability(prob::Vector{T}) where {T <: Number}
    # exceptions (negative pdo, etc)
    # will be added here later
    prob / sum(prob)
end

function conditional_probability(peps::PEPSNetwork, v::Vector{Int}, β::Real)
    i, j = get_coordinates(peps, length(v)+1)
    ∂v = generate_boundary(peps, v, (i, j))

    W = MPO(peps, i, β)
    ψ = MPS(peps, i+1, β)

    L = left_env(ψ, ∂v[1:j-1])
    R = right_env(ψ, W, ∂v[j+2:peps.ncols+1])
    A = peps_tensor(peps, i, j, β)

    l, u = ∂v[j:j+1]
    M = ψ[j]
    Ã = A[l, u, :, :, :]
    @tensor prob[σ] := L[x] * M[x, d, y] *
                       Ã[r, d, σ] * R[y, r] order = (x, d, r, y)
    _normalize_probability(prob)
end

function bond_energy(network, u::NTuple{2, Int}, v::NTuple{2, Int}, σ::Int)
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

function update_energy(network::AbstractGibbsNetwork, σ::Vector{Int})
    i, j = get_coordinates(network, length(σ)+1)

    σkj = _get_local_state(network, σ, (i-1, j))
    σil = _get_local_state(network, σ, (i, j-1))

    bond_energy(network, (i, j), (i, j-1), σil) +
    bond_energy(network, (i, j), (i-1, j), σkj) +
    local_energy(network, (i, j))
end

#TODO: translate this into rotations and reflections
function peps_indices(m::Int, n::Int, origin::Symbol=:NW)
    @assert origin ∈ (:NW, :WN, :NE, :EN, :SE, :ES, :SW, :WS)

    ind = Dict()
    if origin == :NW
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (i - 1) * n + j) end
    elseif origin == :WN
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (j - 1) * n + i) end
    elseif origin == :NE
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (i - 1) * n + (n + 1 - j)) end
    elseif origin == :EN
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (j - 1) * n + (n + 1 - i)) end
    elseif origin == :SE
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (m - i) * n + (n + 1 - j)) end
    elseif origin == :ES
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (m - j) * n + (n + 1 - i)) end
    elseif origin == :SW
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (m - i) * n + j) end
    elseif origin == :WS
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (m - j) * n + i) end
    end

    if origin ∈ (:NW, :NE, :SE, :SW)
        i_max, j_max = m, n
    else
        i_max, j_max = n, m
    end

    for i ∈ 0:i_max+1
        push!(ind, (i, 0) => 0)
        push!(ind, (i, j_max + 1) => 0)
    end

    for j ∈ 0:j_max+1
        push!(ind, (0, j) => 0)
        push!(ind, (i_max + 1, j) => 0)
    end

    ind, i_max, j_max
end
