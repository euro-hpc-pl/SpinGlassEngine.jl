export PegasusNetwork
export pegasus_lattice
export projectors_with_fusing, node_index_with_fusing, compress
export MPO_with_fusing, boundary_at_splitting_node, MPS_with_fusing, conditional_probability, node_from_index, update_energy

function pegasus_lattice(m, n)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    lg = LabelledGraph(labels, grid((m, n)))
    for i in 1:m-1, j in 1:n-1
        add_edge!(lg, (i, j), (i+1, j+1))
    end
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


@memoize Dict function _right_env(peps::PegasusNetwork, i::Int, ∂v::Vector{Int})
    W = MPO_with_fusing(peps, i)
    ψ = MPS_with_fusing(peps, i+1)
    right_env(ψ, W, ∂v)
end

@memoize Dict function _left_env(peps::PegasusNetwork, i::Int, ∂v::Vector{Int})
    ψ = MPS_with_fusing(peps, i+1)
    left_env(ψ, ∂v)
end


function projectors_with_fusing(network::PegasusNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    projs_left = projector.(Ref(network), Ref(vertex), ((i, j-1), (i-1, j-1)))
    pt, pb = projector.(Ref(network), Ref(vertex), ((i-1, j), (i+1, j)))
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
    W = MPO(T, 2 * peps.ncols)
    trr_old = ones(1,1)
    trrd_old = ones(1,1)

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
        v = build_tensor(peps, (i-1, j), (i, j)) ###

        @tensor B[l, u, r, d] := v[u, ũ] * BB[l, ũ, r, d] ####

        W[2*j] = B
        
        h = build_tensor(peps, (i, j-1), (i, j))
        NW = build_tensor(peps, (i-1, j-1), (i, j))

        #@cast C[l, u, r, d] := reduce(x, y, ũ) h[y, x] * trl[l, x] * trlu[l, ũ] * trr_old[r, y] * trrd_old[r, d] * NW[u, ũ]
            
        @tensor C1[l, r] := h[y, x] * trl[l, x] *  trr_old[r, y]      
        @tensor C2[l, u] :=  trlu[l, ũ] * NW[u, ũ]
        @cast C[r, u, l, d] |= C1[l, r] * C2[l, u] * trrd_old[r, d] 

        W[2*j-1] = C

        trr_old = trr  
        trrd_old = trrd 
    end
    W
end

@memoize Dict MPO_with_fusing(
    peps::PegasusNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) = MPO_with_fusing(Float64, peps, i, states_indices)

function compress(
    ψ::AbstractMPS,
    peps::PegasusNetwork;
)
    if bond_dimension(ψ) < peps.bond_dim return ψ end
    SpinGlassTensors.compress(ψ, peps.bond_dim, peps.var_tol, peps.sweeps)
end

node_index_with_fusing(peps::PegasusNetwork, node::NTuple{2, Int}) = peps.ncols * (node[1] - 1) + node[2]

_mod_wo_zero_with_fusing(k, m) = k % m == 0 ? m : k % m

iteration_order(peps::PegasusNetwork) = [(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols]

node_from_index(peps::PegasusNetwork, index::Int) =
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero_with_fusing(index, peps.ncols))



function boundary_at_splitting_node(peps::PegasusNetwork, node::NTuple{2, Int})
    i, j = node
    vcat([
        [
            [((i, k-1), (i+1, k)), ((i, k), (i+1, k))] for k ∈ 1:j-1
        ]...,
        [
            ((i, j-1), (i, j), (i+1, j)) # TODO: second element responsible for fusion
        ]...,
        [
            [((i-1, k-1), (i, k)), ((i-1, k), (i, k))] for k ∈ j:peps.ncols
        ]...
    ]...
    )

end

function compress(
    ψ::AbstractMPS,
    peps::PegasusNetwork;
)
    if bond_dimension(ψ) < peps.bond_dim return ψ end
    SpinGlassTensors.compress(ψ, peps.bond_dim, peps.var_tol, peps.sweeps)
end

@memoize Dict function MPS_with_fusing(
    peps::PegasusNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
)
    if i > peps.nrows return IdentityMPS() end
    W = MPO_with_fusing(peps, i, states_indices)
    ψ = MPS_with_fusing(peps, i+1, states_indices)
    compress(W * ψ, peps)
end


function conditional_probability(peps::PegasusNetwork, v::Vector{Int},
    )
    
        i, j = node_from_index(peps, length(v)+1)
        ∂v = generate_boundary_states_with_fusing(peps, v, (i, j))
        W = MPO_with_fusing(peps, i)
        ψ = MPS_with_fusing(peps, i+1)

        L = _left_env(peps, i, ∂v[1:2*j-2])
        R = _right_env(peps, i, ∂v[2*j+2:peps.ncols*2+1])
        A, _, _ = build_tensor_with_fusing(peps, (i, j))
        v = build_tensor(peps, (i-1, j), (i, j)) 
        
        X = W[2*j-1]

        l, d, u = ∂v[2*j-1:2*j+1]
        MX = ψ[2*j-1]
        M = ψ[2*j]

        vt = v[u, :]
        @tensor Ã[l, r, d, σ] := A[l, x, r, d, σ] * vt[x]
        #@tensor Ã[l, u, r, d] := vt[u, ũ] * A[l, ũ, r, d]

        Xt = X[l, d, :, :]

        
        @tensor prob[σ] := L[x] * Xt[k, y] * MX[x, y, z] * M[z, l, m] *
                            Ã[k, n, l, σ] * R[m, n] order = (x, y, z, k, l, m, n)
    
        _normalize_probability(prob)
    end


function update_energy(network::PegasusNetwork, σ::Vector{Int})
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    bond_energy(network, (i, j), (i-1, j-1), local_state_for_node(network, σ, (i-1, j-1))) +
    local_energy(network, (i, j))
end