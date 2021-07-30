export 
    FusedNetwork, 
    projectors_with_fusing, 
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


    function FusedNetwork(
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
        ng = cross_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end
        new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim, var_tol, sweeps)
    end
end


function projectors_with_fusing(network::FusedNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    projs_left = projector.(
                    Ref(network), Ref(vertex), ((i+1, j-1), (i, j-1), (i-1, j-1))
                )
    pt, pb = projector.(
                    Ref(network), Ref(vertex), ((i-1, j), (i+1, j))
            )
    projs_right = projector.(
                    Ref(network), Ref(vertex), ((i+1, j+1), (i, j+1), (i-1, j+1))
                )

    pl, tl_blt = fuse_projectors(projs_left)
    pr, tr_brt = fuse_projectors(projs_right)

    (pl, pt, pr, pb), tl_blt, tr_brt
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

function SpinGlassTensors.MPO(::Type{T},
    peps::FusedNetwork,
    i::Int,
    pos::Symbol
) where {T <: Number}
    W = MPO(T, 2 * peps.ncols)
    di = pos == :up ? 1 : 0

    for j ∈ 1:peps.ncols
        v = build_tensor(peps, (i-di, j), (i-di+1, j))
        @cast A[_, u, _, d] := v[u, d]
        W[2*j] = A
    end

    # W[2*j-1] = TBW (@Ania && Tomek)
    W
end

function SpinGlassTensors.MPO(::Type{T},
    peps::FusedNetwork,
    i::Int
) where {T <: Number}

    W = MPO(T, 2 * peps.ncols)

    for j ∈ 1:peps.ncols
        neighbours1 = ((i+1, j), (i, j), (i-1, j))
        prl = projector.(Ref(peps), Ref((i, j-1)), neighbours1)
        _, (p_lb, p_l, p_lt) = fuse_projectors(prl)

        neighbours2 = ((i+1, j-1), (i, j-1), (i-1, j-1))
        prr = projector.(Ref(peps), Ref((i, j)), neighbours2)
        _, (p_rb, p_r, p_rt) = fuse_projectors(prr)

        Nh = build_tensor(peps, (i, j-1), (i, j))
        NW = build_tensor(peps, (i-1, j-1), (i, j))
        NE = build_tensor(peps, (i-1, j), (i, j-1))

        @tensor B1[l, r] := p_l[l, x] * Nh[x, y] * p_r[r, y]    
        @tensor B2[l, u] := p_lt[l, ũ] * NE[ũ, u]
        @tensor B3[r, u] := p_rt[r, ũ] * NW[u, ũ]

        @cast B[l, (ũ, u), r, (d̃, d)] |= B1[l, r] * B2[l, u] * p_rb[r, d] * 
                                         B3[r, ũ] * p_lb[l, d̃]
        W[2*j-1] = B
        
        A = build_tensor(peps, (i, j))
        B = dropdims(sum(A, dims=5), dims=5)

        v = build_tensor(peps, (i-1, j), (i, j))
        @tensor C[l, u, r, d] := v[u, ũ] * B[l, ũ, r, d] 
        W[2*j] = C   
    end
    W
end


@memoize Dict SpinGlassTensors.MPO(
    peps::FusedNetwork,
    i::Int
) = SpinGlassTensors.MPO(Float64, peps, i)


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

# to be removed
function initialize_MPS(peps::AbstractGibbsNetwork{S, T}, v::Vector{Int}) where {S, T}
    i, j = node_from_index(peps, length(v)+1)
    (i, j), MPO(peps, i), MPS(peps, i+1), generate_boundary_states(peps, v, (i, j))
end


function conditional_probability(peps::FusedNetwork, v::Vector{Int})   
    i, j = node_from_index(peps, length(v)+1)

    W = MPO(peps, i)
    ψ = MPS(peps, i+1)

    ∂v = generate_boundary_states(peps, v, (i, j))

    L = _left_env(peps, i, ∂v[1:2*j-2])
    R = _right_env(peps, i, ∂v[2*j+2:peps.ncols*2+1])
    A = build_tensor(peps, (i, j))
        
    X, MX, M = W[2*j-1], ψ[2*j-1], ψ[2*j]

    l, d, u = ∂v[2*j-1:2*j+1]
    
    ev = build_tensor(peps, (i-1, j), (i, j)) 
    vt = ev[u, :]

    @tensor Ã[l, r, d, σ] := A[l, x, r, d, σ] * vt[x]
    Xt = @view X[l, d, :, :]
        
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