export PegasusNetwork
export pegasus_lattice
export projectors_with_fusing
export MPO_with_fusing

function pegasus_lattice(m, n)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    lg = LabelledGraph(labels, grid((m, n)))
    #add_edge!(lg, 1, 3)
    add_edge!(lg, (1, 1), (2, 2))
    add_edge!(lg, (1, 2), (2, 3))
    lg
end


struct PegasusNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
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


    function PegasusNetwork(
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
        ng = pegasus_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end
        new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim, var_tol, sweeps)
    end
end


function projectors_with_fusing(network::PegasusNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    projs_left = projector.(Ref(network), Ref(vertex), ((i, j-1), (i-1, j-1)))
    pt, pb = projector.(Ref(network), Ref(vertex), ((i-1, j), (i, j)))
    projs_right = projector.(Ref(network), Ref(vertex), ((i, j+1), (i+1, j+1)))
    # trl, trr ?
    pl, trl = fuse_projectors(projs_left)
    pr, trr = fuse_projectors(projs_right)

    (pl, pt, pr, pb), trl, trr
end


#@memoize Dict peps_tensor(peps::PEPSNetwork, i::Int, j::Int) = peps_tensor(Float64, peps, i, j)

#to be updated to include Pegasus

function MPO_with_fusing(::Type{T},
    peps::PegasusNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) where {T <: Number}
    W = MPO(T, 2 * peps.ncols - 1)
    trr_old = 0
    trrd_old = 0

    for j ∈ 1:peps.ncols
        # from peps_tensor
        A, (trl, trlu), (trr, trrd)  = build_tensor_with_fusing(peps, (i, j))

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

            #@cast C[l, u, r, d] := reduce(x, y, ũ) h[y, x] * trl[l, x] * trlu[l, ũ] * trr_old[r, y] * trrd_old[r, d] * NW[u, ũ]
            
            @tensor C1[l, r] := h[y, x] * trl[l, x] *  trr_old[r, y]      
            @tensor C2[l, u] :=  trlu[l, ũ] * NW[u, ũ]
            @cast C[l, u, r, d] |= C1[l, r] * C2[l, u] * trrd_old[r, d] 

            W[2*j-2] = C
        end
        trr_old = trr  
        trrd_old = trrd 
    end
    W
end