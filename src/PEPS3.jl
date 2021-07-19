export NNNNetwork
export nnn_lattice
export projectors_with_fusing
export MPO_with_fusing

function nnn_lattice(m, n)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    lg = LabelledGraph(labels, grid((m, n)))
    for i in 1:m-1, j in 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
        add_edge!(lg, (i+1, j), (i, j+1))
    end
    lg
end


struct NNNNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
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


    function NNNNetwork(
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
        ng = nnn_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end
        new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim, var_tol, sweeps)
    end
end


function projectors_with_fusing(network::NNNNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    projs_left = projector.(Ref(network), Ref(vertex), ((i+1, j-1), (i, j-1), (i-1, j-1)))
    pt, pb = projector.(Ref(network), Ref(vertex), ((i-1, j), (i+1, j)))
    projs_right = projector.(Ref(network), Ref(vertex), ((i+1, j+1), (i, j+1), (i-1, j+1)))

    # trl, trr ?
    pl, tl_blt = fuse_projectors(projs_left)
    pr, tr_brt = fuse_projectors(projs_right)

    (pl, pt, pr, pb), tl_blt, tr_brt
end


#@memoize Dict peps_tensor(peps::PEPSNetwork, i::Int, j::Int) = peps_tensor(Float64, peps, i, j)

#to be updated to include Pegasus

function MPO_with_fusing(::Type{T},
    peps::NNNNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) where {T <: Number}
    W = MPO(T, 2 * peps.ncols - 1)
    p_rr_old = 0
    p_rt_old = 0
    p_rb_old = 0

    for j ∈ 1:peps.ncols
        # from peps_tensor
        A, (p_lb, p_ll, p_lt), (p_rb, p_rr, p_rt)  = build_tensor_with_fusing(peps, (i, j))

        v = get(states_indices, peps.vertex_map((i, j)), nothing)
        if v !== nothing
            BB = A[:, :, :, :, v]
        else
            BB = dropdims(sum(A, dims=5), dims=5)
            #@reduce B[l, u, r, d] |= sum(σ) A[l, u, r, d, σ]
        end
        # include energy
        v = build_tensor(peps, (i-1, j), (i, j))

        @tensor B[l, u, r, d] := v[u, ũ] * BB[l, ũ, r, d]

        W[2*j-1] = B
        
        if j > 1
            h = build_tensor(peps, (i, j-1), (i, j))
            NW = build_tensor(peps, (i-1, j-1), (i, j))
            NE = build_tensor(peps, (i-1, j), (i, j-1))

            #@cast C[l, u, r, d] := reduce(x, y, ũ) h[y, x] * trl[l, x] * trlu[l, ũ] * trr_old[r, y] * trrd_old[r, d] * NW[u, ũ]
            @tensor C1[l, r] := p_rr_old[l, x] * h[x, y] * p_ll[r, y]    
            @tensor C2[l, u] :=  p_rt_old[l, ũ] * NE[ũ, u]
            @tensor C3[r, uu] :=  p_lt[r, ũ] * NW[uu, ũ]
            @cast C[l, (uu, u), r, (dd, d)] |= C1[l, r] * C2[l, u] * p_lb[r, d] * C3[r, uu] * p_rb_old[l, dd]
            W[2*j-2] = C
        end
        p_rb_old = p_rb 
        p_rt_old = p_rt
        p_rr_old = p_rr
    end
    W
end


function boundary_at_splitting_node(peps::NNNNetwork, node::NTuple{2, Int})
    i, j = node
    [
        [((i, k), (i+1, k)) for k ∈ 1:j-1]...,
        ((i, j-1), (i, j)),
        [((i-1, k), (i, k)) for k ∈ j:peps.ncols]...,
        [((i-1, k), (i, k)) for k ∈ 1:j-1]...,
        [((i, k), (i+1, k)) for k ∈ j:peps.ncols]...
    ]
end


function generate_boundary_states_with_fusing(
    network::NNNNetwork,
    σ::Vector{Int},
    node::S 
) where {S, T}
    [
        generate_boundary_state_with_fusing(network, v, w, local_state_for_node(network, σ, v))
        for (v, w) ∈ boundary_at_splitting_node(network, node)
    ]
end