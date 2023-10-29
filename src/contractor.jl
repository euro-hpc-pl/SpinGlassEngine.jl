export
       SVDTruncate,
       MPSAnnealing,
       Zipper,
       MpoLayers,
       MpsParameters,
       MpsContractor,
       NoUpdate,
       GaugeStrategy,
       GaugeStrategyWithBalancing,
       clear_memoize_cache,
       clear_memoize_cache_after_row,
       mpo,
       mps_top,
       mps,
       mps_top_approx,
       mps_approx,
       update_gauges!,
       sweep_gauges!,
       update_gauges_with_balancing!,
       boundary_states,
       dressed_mps,
       error_measure,
       conditional_probability,
       update_energy,
       boundary,
       local_state_for_node,
       boundary_indices,
       layout,
       sparsity,
       strategy,
       left_env,
       right_env

abstract type AbstractContractor end
abstract type AbstractStrategy end
abstract type AbstractGauge end

struct SVDTruncate <: AbstractStrategy end
struct MPSAnnealing <: AbstractStrategy end
struct Zipper <: AbstractStrategy end
struct GaugeStrategyWithBalancing <: AbstractGauge end
struct GaugeStrategy <: AbstractGauge end
struct NoUpdate <: AbstractGauge end

"""
$(TYPEDSIGNATURES)

A struct representing different layers of a Matrix Product Operator (MPO) used in contraction algorithms.

# Fields
- `main::Dict{Site, Sites}`: A dictionary mapping sites to the main layers of the MPO.
- `dress::Dict{Site, Sites}`: A dictionary mapping sites to the dress layers of the MPO.
- `right::Dict{Site, Sites}`: A dictionary mapping sites to the right layers of the MPO.

The `MpoLayers` struct distinguishes the various layers of an MPO, which is often used in tensor network contraction algorithms. MPOs are commonly employed in quantum many-body physics and condensed matter physics to represent operators acting on quantum states in a factorized form.
"""
struct MpoLayers
    main::Dict{Site, Sites}
    dress::Dict{Site, Sites}
    right::Dict{Site, Sites}
end

"""
$(TYPEDSIGNATURES)

A struct representing control parameters for the MPO-MPS (Matrix Product Operator - Matrix Product State) scheme used to contract a PEPS (Projected Entangled Pair States) network.

# Fields
- `bond_dimension::Int`: The maximum bond dimension to be used during contraction.
- `variational_tol::Real`: The tolerance for the variational solver used in optimization.
- `max_num_sweeps::Int`: The maximum number of sweeps to perform during contraction.
- `tol_SVD::Real`: The tolerance used in singular value decomposition (SVD) operations.
- `iters_svd::Int`: The number of iterations to perform in SVD computations.
- `iters_var::Int`: The number of iterations for variational optimization.
- `Dtemp_multiplier::Int`: A multiplier for the bond dimension when temporary bond dimensions are computed.
- `method::Symbol`: The contraction method to use (e.g., `:psvd_sparse`).

The `MpsParameters` struct encapsulates various control parameters that influence the behavior and accuracy of the MPO-MPS contraction scheme used for PEPS network calculations.
"""
struct MpsParameters
    bond_dimension::Int
    variational_tol::Real
    max_num_sweeps::Int
    tol_SVD::Real
    iters_svd::Int
    iters_var::Int
    Dtemp_multiplier::Int
    method::Symbol

    MpsParameters(
        bd = typemax(Int),
        ϵ = 1E-8,
        sw = 4,
        ts = 1E-16,
        is = 1,
        iv = 1,
        dm = 2,
        m = :psvd_sparse
    ) = new(bd, ϵ, sw, ts, is, iv, dm, m)
end

"""
$(TYPEDSIGNATURES)
A function that provides the layout used to construct the PEPS (Projected Entangled Pair States) network.

# Arguments
- `net::PEPSNetwork{T, S}`: The PEPS network for which the layout is provided.

# Returns
- The layout type `T` used to construct the PEPS network.

The `layout` function returns the layout type used in the construction of a PEPS network. This layout type specifies the geometric arrangement and sparsity pattern of the tensors in the PEPS network.
"""
layout(net::PEPSNetwork{T, S}) where {T, S} = T

"""
$(TYPEDSIGNATURES)
A function that provides the sparsity used to construct the PEPS (Projected Entangled Pair States) network.

# Arguments
- `net::PEPSNetwork{T, S}`: The PEPS network for which the sparsity is provided.

# Returns
- The sparsity type `S` used to construct the PEPS network.

The `sparsity` function returns the sparsity type used in the construction of a PEPS network. This sparsity type specifies the pattern of zero elements in the tensors of the PEPS network, which can affect the computational efficiency and properties of the network.
"""
sparsity(net::PEPSNetwork{T, S}) where {T, S} = S

"""
$(TYPEDSIGNATURES)
A mutable struct representing a contractor for contracting a PEPS (Projected Entangled Pair States) network using the MPO-MPS (Matrix Product Operator - Matrix Product State) scheme.

# Fields
- `peps::PEPSNetwork{T, S}`: The PEPS network to be contracted.
- `betas::Vector{<:Real}`: A vector of inverse temperatures (β) for thermal contraction.
- `graduate_truncation::Symbol`: The truncation method to use for "graduating" the bond dimensions.
- `params::MpsParameters`: Control parameters for the MPO-MPS contraction.
- `layers::MpoLayers`: The layers of the MPO (Matrix Product Operator).
- `statistics::Dict{Vector{Int}, <:Real}`: Statistics collected during the contraction process.
- `nodes_search_order::Vector{Node}`: The order in which nodes are searched during contraction.
- `node_outside::Node`: A node representing outside connections.
- `node_search_index::Dict{Node, Int}`: A mapping of nodes to their search indices.
- `current_node::Node`: The current node being processed during contraction.
- `onGPU::Bool`: A flag indicating whether the contraction is performed on a GPU.

The `MpsContractor` struct defines the contractor responsible for contracting a PEPS network using the MPO-MPS scheme.
It encapsulates various components and settings required for the contraction process.
"""
mutable struct MpsContractor{T <: AbstractStrategy, R <: AbstractGauge} <: AbstractContractor
    peps::PEPSNetwork{T, S} where {T, S}
    betas::Vector{<:Real}
    graduate_truncation::Symbol
    params::MpsParameters
    layers::MpoLayers
    statistics#::Dict{Vector{Int}, <:Real}
    nodes_search_order::Vector{Node}
    node_outside::Node
    node_search_index::Dict{Node, Int}
    current_node::Node
    onGPU::Bool

    function MpsContractor{T, R}(net, βs, graduate_truncation::Symbol, params; onGPU=true) where {T, R}
        ml = MpoLayers(layout(net), net.ncols)
        stat = Dict()
        ord, node_out = nodes_search_order_Mps(net)
        enum_ord = Dict(node => i for (i, node) ∈ enumerate(ord))
        node = ord[begin]
        new(net, βs, graduate_truncation, params, ml, stat, ord, node_out, enum_ord, node, onGPU)
    end
end

"""
$(TYPEDSIGNATURES)
Get the strategy used to contract the PEPS network.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.

# Returns
- `T`: The strategy used for network contraction.
"""
strategy(ctr::MpsContractor{T}) where {T} = T

"""
$(TYPEDSIGNATURES)
Construct and memoize a Matrix Product Operator (MPO) for a given set of layers.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `layers::Dict{Site, Sites}`: A dictionary mapping sites to their corresponding layers.
- `r::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMpo`: The constructed MPO for the specified layers.

This function constructs an MPO by iterating through the specified layers and assembling the corresponding tensors. The resulting MPO is memoized for efficient reuse.
"""
@memoize Dict function mpo(
    ctr::MpsContractor{T}, layers::Dict{Site, Sites}, r::Int, indβ::Int
) where T <: AbstractStrategy
    mpo = Dict{Site, MpoTensor{Float64}}() # Float64 - for now
    for (site, coordinates) ∈ layers
        lmpo = TensorMap{Float64}()  # Float64 - for now
        for dr ∈ coordinates
            ten = tensor(ctr.peps, PEPSNode(r + dr, site), ctr.betas[indβ])
            push!(lmpo, dr => ten)
        end
        push!(mpo, site => MpoTensor(lmpo))
    end
    ctr.onGPU ? move_to_CUDA!(QMpo(mpo)) : QMpo(mpo)
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the top Matrix Product State (MPS) using Singular Value Decomposition (SVD) for a given row.

# Arguments
- `ctr::MpsContractor{SVDTruncate}`: The MpsContractor object representing the PEPS network contraction with SVD truncation.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMps`: The constructed top MPS for the specified row.

This function constructs the top MPS using SVD for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, truncation, and compression steps as needed based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps_top(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i < 1
        W = mpo(ctr, ctr.layers.main, 1, indβ)
        return IdentityQMps(Float64, local_dims(W, :up); onGPU=ctr.onGPU) # F64 for now
    end

    ψ = mps_top(ctr, i-1, indβ)
    W = transpose(mpo(ctr, ctr.layers.main, i, indβ))
    ψ0 = dot(W, ψ)

    canonise!(ψ0, :right)
    if ctr.graduate_truncation == :graduate_truncate
        canonise_truncate!(ψ0, :left, Dcut * 2, tolS / 2)
        variational_sweep!(ψ0, W, ψ, Val(:right))
    end
    canonise_truncate!(ψ0, :left, Dcut, tolS)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) Matrix Product State (MPS) using Singular Value Decomposition (SVD) for a given row.

# Arguments
- `ctr::MpsContractor{SVDTruncate}`: The MpsContractor object representing the PEPS network contraction with SVD truncation.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMps`: The constructed (bottom) MPS for the specified row.

This function constructs the (bottom) MPS using SVD for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, truncation, and compression steps as needed based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(Float64, local_dims(W, :down); onGPU=ctr.onGPU) # Float64 fror now
    end

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    ψ0 = dot(W, ψ)
    canonise!(ψ0, :right)
    if ctr.graduate_truncation == :graduate_truncate
        canonise_truncate!(ψ0, :left, Dcut * 2, tolS / 2)
        variational_sweep!(ψ0, W, ψ, Val(:right))
    end
    canonise_truncate!(ψ0, :left, Dcut, tolS)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end


"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) Matrix Product State (MPS) approximation using Singular Value Decomposition (SVD) for a given row.

# Arguments
- `ctr::MpsContractor{SVDTruncate}`: The MpsContractor object representing the PEPS network contraction with SVD truncation.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMps`: The constructed (bottom) MPS approximation for the specified row.

This function constructs the (bottom) MPS approximation using SVD for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, and truncation steps based on the specified parameters in `ctr.params`. The resulting MPS approximation is memoized for efficient reuse.
"""
@memoize Dict function mps_approx(ctr::MpsContractor{SVDTruncate}, i::Int, indβ::Int)
    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(Float64, local_dims(W, :down); onGPU=ctr.onGPU) # F64 for now
    end

    W = mpo(ctr, ctr.layers.main, i, indβ)
    ψ = IdentityQMps(Float64, local_dims(W, :down); onGPU=ctr.onGPU) # F64 for now

    ψ0 = dot(W, ψ)
    truncate!(ψ0, :left, ctr.params.bond_dimension)
    ψ0
end

#=
function (ctr::MpsContractor)(peps::PEPSNetwork, ...., :mps_top)

for ctr ∈ [ctr_1, ctr_2]
    mpo_1 = ctr_1(peps, ..., :mpo)
    mpo_2 = ctr_2(peps, ..., :mpo)

    #@nexprs 2 k k -> mpo_k = ctr_k(peps, ..., :mpo)
end
=#

"""
$(TYPEDSIGNATURES)

Construct and memoize the top Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row.

# Arguments
- `ctr::MpsContractor{Zipper}`: The MpsContractor object representing the PEPS network contraction with the Zipper method.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMps`: The constructed top MPS using the Zipper method for the specified row.

This function constructs the top Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, and truncation steps based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps_top(ctr::MpsContractor{Zipper}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps
    iters_svd = ctr.params.iters_svd
    iters_var = ctr.params.iters_var
    Dtemp_multiplier = ctr.params.Dtemp_multiplier
    method = ctr.params.method
    if i < 1
        W = mpo(ctr, ctr.layers.main, 1, indβ)
        return IdentityQMps(Float64, local_dims(W, :up); onGPU=ctr.onGPU) # F64 for now
    end

    ψ = mps_top(ctr, i-1, indβ)
    W = transpose(mpo(ctr, ctr.layers.main, i, indβ))

    canonise!(ψ, :left)
    ψ0 = zipper(W, ψ; method=method, Dcut=Dcut, tol=tolS, iters_svd=iters_svd,
                iters_var=iters_var, Dtemp_multiplier = Dtemp_multiplier)
    canonise!(ψ0, :left)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row.

# Arguments
- `ctr::MpsContractor{Zipper}`: The MpsContractor object representing the PEPS network contraction with the Zipper method.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMps`: The constructed (bottom) MPS using the Zipper method for the specified row.

This function constructs the (bottom) Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, and truncation steps based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps(ctr::MpsContractor{Zipper}, i::Int, indβ::Int)
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps
    iters_svd = ctr.params.iters_svd
    iters_var = ctr.params.iters_var
    Dtemp_multiplier = ctr.params.Dtemp_multiplier
    method = ctr.params.method

    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        ψ0 = IdentityQMps(Float64, local_dims(W, :down); onGPU=ctr.onGPU) # Float64 for now
    else
        ψ = mps(ctr, i+1, indβ)
        W = mpo(ctr, ctr.layers.main, i, indβ)
        canonise!(ψ, :left)
        ψ0 = zipper(W, ψ; method=method, Dcut=Dcut, tol=tolS, iters_svd=iters_svd,
                    iters_var=iters_var, Dtemp_multiplier=Dtemp_multiplier)
        canonise!(ψ0, :left)
        variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    end
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) top Matrix Product State (MPS) using the Annealing method for a given row.

# Arguments
- `ctr::MpsContractor{MPSAnnealing}`: The MpsContractor object representing the PEPS network contraction with the Annealing method.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMps`: The constructed (bottom) top MPS using the Annealing method for the specified row.

This function constructs the (bottom) top Matrix Product State (MPS) using the Annealing method for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing variational compression steps based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps_top(ctr::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
    if i < 1
        W = mpo(ctr, ctr.layers.main, 1, indβ)
        return IdentityQMps(Float64, local_dims(W, :up); onGPU=ctr.onGPU) # F64 for now
    end

    ψ = mps_top(ctr, i-1, indβ)
    W = transpose(mpo(ctr, ctr.layers.main, i, indβ))

    if indβ > 1
        ψ0 = mps_top(ctr, i, indβ-1)
    else
        ψ0 = IdentityQMps(Float64, local_dims(W, :up), ctr.params.bond_dimension; onGPU=ctr.onGPU) # F64 for now
        # ψ0 = IdentityQMps(Float64, local_dims(W, :down), ctr.params.bond_dimension) # F64 for now
        canonise!(ψ0, :left)
    end
    variational_compress!(
        ψ0,
        W,
        ψ,
        ctr.params.variational_tol,
        ctr.params.max_num_sweeps,
    )
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) Matrix Product State (MPS) using the Annealing method for a given row.

# Arguments
- `ctr::MpsContractor{MPSAnnealing}`: The MpsContractor object representing the PEPS network contraction with the Annealing method.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta values.

# Returns
- `QMps`: The constructed (bottom) MPS using the Annealing method for the specified row.

This function constructs the (bottom) Matrix Product State (MPS) using the Annealing method for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing variational compression steps based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps(ctr::MpsContractor{MPSAnnealing}, i::Int, indβ::Int)
    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows, indβ)
        return IdentityQMps(Float64, local_dims(W, :down); onGPU=ctr.onGPU) # F64 for now
    end

    ψ = mps(ctr, i+1, indβ)
    W = mpo(ctr, ctr.layers.main, i, indβ)

    if indβ > 1
        ψ0 = mps(ctr, i, indβ-1)
    else
        ψ0 = IdentityQMps(Float64, local_dims(W, :up), ctr.params.bond_dimension; onGPU=ctr.onGPU) # F64 for now
        canonise!(ψ0, :left)
    end

    variational_compress!(
        ψ0,
        W,
        ψ,
        ctr.params.variational_tol,
        ctr.params.max_num_sweeps
    )
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct dressed Matrix Product State (MPS) for a given row and strategy.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `i::Int`: The current row index.

# Returns
- `QMps`: The constructed dressed MPS for the specified row.

This function constructs the dressed Matrix Product State (MPS) for a given row in the PEPS network contraction using the specified strategy. It internally calculates the length of the `ctr.betas` vector and then calls `dressed_mps(ctr, i, length(ctr.betas))` to construct the dressed MPS with the given parameters.
"""
function dressed_mps(ctr::MpsContractor{T}, i::Int) where T <: AbstractStrategy
    dressed_mps(ctr, i, length(ctr.betas))
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) dressed Matrix Product State (MPS) for a given row and strategy.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `i::Int`: The current row index.
- `indβ::Int`: The index of the beta parameter vector used for construction.

# Returns
- `QMps`: The constructed dressed MPS for the specified row and strategy.

This function constructs the dressed Matrix Product State (MPS) for a given row in the PEPS network contraction using the specified strategy and memoizes the result for future use. It takes into account the beta parameter index `indβ` and internally calls other functions such as `mps` and `mpo` to construct the dressed MPS. Additionally, it normalizes the MPS tensors to ensure numerical stability.

Note: The memoization ensures that the dressed MPS is only constructed once for each combination of arguments and is reused when needed.
"""
@memoize Dict function dressed_mps(
    ctr::MpsContractor{T}, i::Int, indβ::Int
) where T <: AbstractStrategy

    ψ = mps(ctr, i+1, indβ)
    delete!(Memoization.caches[mps], ((ctr, i+1, indβ), ()))
    if ctr.onGPU
        ψ = move_to_CUDA!(ψ)
    end
    W = mpo(ctr, ctr.layers.dress, i, indβ)
    ϕ = dot(W, ψ)
    for j ∈ ϕ.sites
        nrm = maximum(abs.(ϕ[j]))
        if !iszero(nrm) ϕ[j] ./= nrm end
    end
    ϕ
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) the right environment tensor for a given node in the PEPS network contraction.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `i::Int`: The current row index.
- `∂v::Vector{Int}`: A vector representing the partial environment configuration.
- `indβ::Int`: The index of the beta parameter vector used for construction.

# Returns
- `Array{Float64,2}`: The constructed right environment tensor for the specified node.

This function constructs the right environment tensor for a given node in the PEPS network contraction using the specified strategy and memoizes the result for future use. It takes into account the beta parameter index `indβ` and internally calls other functions such as `dressed_mps` and `mpo` to construct the right environment tensor. Additionally, it normalizes the right environment tensor to ensure numerical stability.

Note: The memoization ensures that the right environment tensor is only constructed once for each combination of arguments and is reused when needed.
"""
@memoize Dict function right_env(
    ctr::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T <: AbstractStrategy
    l = length(∂v)
    if l == 0 return ctr.onGPU ? CUDA.ones(Float64, 1, 1) : ones(Float64, 1, 1) end

    R̃ = right_env(ctr, i, ∂v[2:l], indβ)
    if ctr.onGPU
        R̃ = CuArray(R̃)
    end
    ϕ = dressed_mps(ctr, i, indβ)
    W = mpo(ctr, ctr.layers.right, i, indβ)
    k = length(ϕ.sites)
    site = ϕ.sites[k-l+1]
    M = W[site]
    B = ϕ[site]

    RR = update_reduced_env_right(R̃, ∂v[1], M, B)

    ls_mps = left_nbrs_site(site, ϕ.sites)
    ls = left_nbrs_site(site, W.sites)

    while ls > ls_mps
        RR = update_reduced_env_right(RR, W[ls].ctr)
        ls = left_nbrs_site(ls, W.sites)
    end
    nmr = maximum(abs.(RR))
    if ~iszero(nmr)
        RR ./= nmr
    end
    if typeof(RR) <: CuArray
        RR = Array(RR)
    end
    RR
end


"""
$(TYPEDSIGNATURES)
Construct (and memoize) the left environment tensor for a given node in the PEPS network contraction.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `i::Int`: The current row index.
- `∂v::Vector{Int}`: A vector representing the partial environment configuration.
- `indβ::Int`: The index of the beta parameter vector used for construction.

# Returns
- `Array{Float64,2}`: The constructed left environment tensor for the specified node.

This function constructs the left environment tensor for a given node in the PEPS network contraction using the specified strategy and memoizes the result for future use. It takes into account the beta parameter index `indβ` and internally calls other functions such as `dressed_mps` to construct the left environment tensor. Additionally, it normalizes the left environment tensor to ensure numerical stability.

Note: The memoization ensures that the left environment tensor is only constructed once for each combination of arguments and is reused when needed.

"""
@memoize Dict function left_env(
    ctr::MpsContractor{T}, i::Int, ∂v::Vector{Int}, indβ::Int
) where T
    l = length(∂v)
    if l == 0 return ctr.onGPU ? CUDA.ones(Float64, 1) : ones(Float64, 1) end

    L̃ = left_env(ctr, i, ∂v[1:l-1], indβ)
    ϕ = dressed_mps(ctr, i, indβ)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]

    @matmul L[x] := sum(α) L̃[α] * M[α, x, $m]
    nmr = maximum(abs.(L))
    iszero(nmr) ? L : L ./ nmr
end

"""
$(TYPEDSIGNATURES)
Clear all memoization caches used by the PEPS network contraction.

This function clears all memoization caches that store previously computed results for various operations and environments in the PEPS network contraction.
Memoization is used to optimize the contraction process by avoiding redundant computations.
Calling this function removes all cached results, which can be useful when you want to free up memory or ensure that the caches are refreshed with updated data.
"""
function clear_memoize_cache()
    Memoization.empty_all_caches!()
end

"""
$(TYPEDSIGNATURES)

Clear memoization caches for specific operations after processing a row.
This function clears the memoization caches for specific operations used in the PEPS network contraction after processing a row.
The cleared operations include `left_env`, `right_env`, `mpo`, and `dressed_mps`. Memoization is used to optimize the contraction process by avoiding redundant computations.
Calling this function allows you to clear the caches for these specific operations, which can be useful when you want to free up memory or ensure that the caches are refreshed with updated data after processing a row in the contraction.
"""
function clear_memoize_cache_after_row()
    Memoization.empty_cache!.((left_env, right_env, mpo, dressed_mps))
end

"""
$(TYPEDSIGNATURES)
Clear memoization cache for specific operations for a given row and index beta.

This function clears the memoization cache for specific operations used in the PEPS network contraction for a given row and index beta (indβ).
The cleared operations include `mps_top`, `mps`, `mpo`, `dressed_mps`, and related operations.
Memoization is used to optimize the contraction process by avoiding redundant computations.
Calling this function allows you to clear the cache for these specific operations for a particular row and index beta, which can be useful when you want to free up memory or ensure that the cache is refreshed with updated data for a specific computation.

# Arguments
- `ctr::MpsContractor{T, S}`: The PEPS network contractor object.
- `row::Site`: The row for which the cache should be cleared.
- `indβ::Int`: The index beta for which the cache should be cleared.

"""
function clear_memoize_cache(ctr::MpsContractor{T, S}, row::Site, indβ::Int) where {T, S}
    for ind ∈ 1:indβ # indbeta a vector?
        for i ∈ row:ctr.peps.nrows
            delete!(Memoization.caches[mps_top], ((ctr, i, ind), ()))
        end
        for i ∈ 1:row+1
            delete!(Memoization.caches[mps], ((ctr, i, ind), ()))
        end
        for i ∈ row:row+2
            cmpo = Memoization.caches[mpo]
            delete!(cmpo, ((ctr, ctr.layers.main, i, ind), ()))
            delete!(cmpo, ((ctr, ctr.layers.dress, i, ind), ()))
            delete!(cmpo, ((ctr, ctr.layers.right, i, ind), ()))
        end

    end
end

"""
$(TYPEDSIGNATURES)
"""
function error_measure(probs)
    if maximum(probs) <= 0 return 2.0 end
    if minimum(probs) < 0 return abs(minimum(probs)) / maximum(abs.(probs)) end
    return 0.0
end

"""
$(TYPEDSIGNATURES)
"""
function sweep_gauges!(
    ctr::MpsContractor{T, GaugeStrategy},
    row::Site,
    indβ::Int,
    tol::Real=1E-4,
    max_sweeps::Int=10
) where T
    clm = ctr.layers.main
    ψ_top = mps_top(ctr, row, indβ)
    ψ_bot = mps(ctr, row + 1, indβ)

    ψ_top = deepcopy(ψ_top)
    ψ_bot = deepcopy(ψ_bot)

    onGPU = ψ_top.onGPU && ψ_bot.onGPU

    gauges = optimize_gauges_for_overlaps!!(ψ_top, ψ_bot, tol, max_sweeps)

    for i ∈ ψ_top.sites
        g = gauges[i]
        g_inv = 1.0 ./ g
        @inbounds n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        @inbounds n_top = PEPSNode(row + clm[i][end], i)
        top = ctr.peps.gauges.data[n_top]
        bot = ctr.peps.gauges.data[n_bot]
        onGPU ? top = CuArray(top) : top
        onGPU ? bot = CuArray(bot) : bot
        g_top = top .* g
        g_bot = bot .* g_inv
        push!(ctr.peps.gauges.data, n_top => g_top, n_bot => g_bot)
    end
    clear_memoize_cache(ctr, row, indβ)
end

"""
$(TYPEDSIGNATURES)
"""
function sweep_gauges!(
    ctr::MpsContractor{T, GaugeStrategyWithBalancing},
    row::Site,
    indβ::Int,
    ) where T
    clm = ctr.layers.main
    ψ_top = mps_top(ctr, row, indβ)
    ψ_bot = mps(ctr, row + 1, indβ)
    ψ_top = deepcopy(ψ_top)
    ψ_bot = deepcopy(ψ_bot)
    for i ∈ ψ_top.sites
        @inbounds n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        @inbounds n_top = PEPSNode(row + clm[i][end], i)
        ρ = overlap_density_matrix(ψ_top, ψ_bot, i)
        _, _, scale = LinearAlgebra.LAPACK.gebal!('S', ρ)
        push!(ctr.peps.gauges.data, n_top => 1.0 ./ scale, n_bot => scale)
    end
    clear_memoize_cache(ctr, row, indβ)
    ψ_top * ψ_bot
end

"""
$(TYPEDSIGNATURES)
"""
function sweep_gauges!(
    ctr::MpsContractor{T, NoUpdate},
    row::Site,
    indβ::Int,
    tol::Real=1E-4,
    max_sweeps::Int=10
) where T

end

"""
$(TYPEDSIGNATURES)
"""
function update_gauges!(
    ctr::MpsContractor{T, S},
    row::Site,
    indβ::Vector{Int},
    ::Val{:down}
) where {T, S}
    for j ∈ indβ, i ∈ 1:row-1
        sweep_gauges!(ctr, i, j)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function update_gauges!(
    ctr::MpsContractor{T, S},
    row::Site,
    indβ::Vector{Int},
    ::Val{:up}
) where {T, S}
    for j ∈ indβ, i ∈ row-1:-1:1
        sweep_gauges!(ctr, i, j)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function conditional_probability(ctr::MpsContractor{S}, w::Vector{Int}) where S
    conditional_probability(layout(ctr.peps), ctr, w)
end

"""
$(TYPEDSIGNATURES)
"""
function update_energy(ctr::MpsContractor{S}, w::Vector{Int}) where S
    update_energy(layout(ctr.peps), ctr, w)
end

function boundary_states(
    ctr::MpsContractor{T}, states::Vector{Vector{Int}}, node::S
) where {T, S}
    boundary_recipe = boundary(ctr, node)
    res = ones(Int, length(states), length(boundary_recipe))
    for (i, node) ∈ enumerate(boundary_recipe)
        @inbounds res[:, i] = boundary_indices(ctr, node, states)
    end
    [res[r, :] for r ∈ 1:size(res, 1)]
end

"""
$(TYPEDSIGNATURES)
"""
function boundary(ctr::MpsContractor{T}, node::Node) where T
    boundary(layout(ctr.peps), ctr, node)
end

"""
$(TYPEDSIGNATURES)
"""
function local_state_for_node(
    ctr::MpsContractor{T}, σ::Vector{Int}, w::S
) where {T, S}
    k = get(ctr.node_search_index, w, 0)
    0 < k <= length(σ) ? σ[k] : 1
end

"""
$(TYPEDSIGNATURES)
"""
function boundary_indices(
    ctr::MpsContractor{T},
    nodes::Union{NTuple{2, S}, Tuple{S, NTuple{N, S}}},
    states::Vector{Vector{Int}}
) where {T, S, N}
    v, w = nodes
    if ctr.peps.vertex_map(v) ∈ vertices(ctr.peps.clustered_hamiltonian)
        @inbounds idx = [σ[ctr.node_search_index[v]] for σ ∈ states]
        return @inbounds projector(ctr.peps, v, w)[idx]
    end
    ones(Int, length(states))
end

"""
$(TYPEDSIGNATURES)

boundary index formed from outer product of two projectors
"""
function boundary_indices(
    ctr::MpsContractor{T},
    nodes::Union{NTuple{4, S}, Tuple{S, NTuple{2, S}, S, NTuple{2, S}}},
    states::Vector{Vector{Int}}
) where {S, T}
    v, w, k, l = nodes
    pv = projector(ctr.peps, v, w)
    i = boundary_indices(ctr, (v, w), states)
    j = boundary_indices(ctr, (k, l), states)
    (j .- 1) .* maximum(pv) .+ i
end

function boundary_indices(
    ctr::MpsContractor{T},
    nodes::Tuple{S, NTuple{2, S}, S, NTuple{2, S}, S, NTuple{2, S}, S, NTuple{2, S}},
    states::Vector{Vector{Int}}
) where {S, T}
    v1, v2, v3, v4, v5, v6, v7, v8 = nodes
    pv1 = projector(ctr.peps, v1, v2)
    pv3 = projector(ctr.peps, v3, v4)
    mm = maximum(pv1) * maximum(pv3)
    i = boundary_indices(ctr, (v1, v2, v3, v4), states)
    j = boundary_indices(ctr, (v5, v6, v7, v8), states)
    (j .- 1) .* mm .+ i
end
