export FusedNetwork
export projectors_with_fusing, boundary_at_splitting_node
export conditional_probability, update_energy

function cross_lattice(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    lg = LabelledGraph(labels, grid((m, n)))
    for i in 1:m-1, j in 1:n-1
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
    projs_left = projector.(Ref(network), Ref(vertex), ((i+1, j-1), (i, j-1), (i-1, j-1)))
    pt, pb = projector.(Ref(network), Ref(vertex), ((i-1, j), (i+1, j)))
    projs_right = projector.(Ref(network), Ref(vertex), ((i+1, j+1), (i, j+1), (i-1, j+1)))

    pl, tl_blt = fuse_projectors(projs_left)
    pr, tr_brt = fuse_projectors(projs_right)

    (pl, pt, pr, pb), tl_blt, tr_brt
end

#@memoize Dict peps_tensor(peps::PEPSNetwork, i::Int, j::Int) = peps_tensor(Float64, peps, i, j)

function SpinGlassTensors.MPO(::Type{T},
    peps::FusedNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) where {T <: Number}

    W = MPO(T, 2 * peps.ncols)
    p_rr_old = ones(1, 1)
    p_rt_old = ones(1, 1)
    p_rb_old = ones(1, 1)

    for j ∈ 1:peps.ncols
        # from peps_tensor
        A, (p_lb, p_ll, p_lt), (p_rb, p_rr, p_rt) = build_tensor(peps, (i, j))

        v = get(states_indices, peps.vertex_map((i, j)), nothing)
        if v !== nothing
            BB = A[:, :, :, :, v]
        else
            BB = dropdims(sum(A, dims=5), dims=5)
        end

        # include energy
        v = build_tensor(peps, (i-1, j), (i, j))
        @tensor B[l, u, r, d] := v[u, ũ] * BB[l, ũ, r, d]
        W[2*j] = B
        
        h = build_tensor(peps, (i, j-1), (i, j))
        NW = build_tensor(peps, (i-1, j-1), (i, j))
        NE = build_tensor(peps, (i-1, j), (i, j-1))

        @tensor C1[l, r] := p_rr_old[l, x] * h[x, y] * p_ll[r, y]    
        @tensor C2[l, u] :=  p_rt_old[l, ũ] * NE[ũ, u]
        @tensor C3[r, uu] :=  p_lt[r, ũ] * NW[uu, ũ]
        @cast C[l, (uu, u), r, (dd, d)] |= C1[l, r] * C2[l, u] * p_lb[r, d] * 
                                           C3[r, uu] * p_rb_old[l, dd]
        W[2*j-1] = C

        p_rb_old = p_rb 
        p_rt_old = p_rt
        p_rr_old = p_rr
    end
    W
end


@memoize Dict SpinGlassTensors.MPO(
    peps::FusedNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) = SpinGlassTensors.MPO(Float64, peps, i, states_indices)


function boundary_at_splitting_node(peps::FusedNetwork, node::NTuple{2, Int})
    i, j = node
    vcat(
        [
            [((i, k-1), (i+1, k), (i, k), (i+1, k-1)), ((i, k), (i+1, k))] for k ∈ 1:j-1
        ]...,
        [
            ((i, j-1), (i, j), (i+1, j)) 
        ]...,
        [
            [((i-1, k-1), (i, k), (i-1, k), (i, k-1)), ((i-1, k), (i, k))] for k ∈ j:peps.ncols
        ]...
    )
end


function conditional_probability(peps::FusedNetwork, v::Vector{Int})   
    i, j = node_from_index(peps, length(v)+1)
    ∂v = generate_boundary_states(peps, v, (i, j))

    W = MPO(peps, i)
    ψ = MPS(peps, i+1)

    L = _left_env(peps, i, ∂v[1:2*j-2])
    R = _right_env(peps, i, ∂v[2*j+2:peps.ncols*2+1])
    A, _, _ = build_tensor(peps, (i, j))
    v = build_tensor(peps, (i-1, j), (i, j)) 
        
    X = W[2*j-1]

    l, d, u = ∂v[2*j-1:2*j+1]
    MX = ψ[2*j-1]
    M = ψ[2*j]

    vt = v[u, :]
    @tensor Ã[l, r, d, σ] := A[l, x, r, d, σ] * vt[x]

    Xt = X[l, d, :, :]
        
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