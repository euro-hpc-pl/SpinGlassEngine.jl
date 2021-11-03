export
    MpsContractor,
    MpoLayers,
    MpsParameters

abstract type AbstractContractor end

struct MpoLayers
    main::Dict
    dress::Dict
    right::Dict
end


struct MpsParameters
    bond_dimension::Int
    variational_tol::Real
    max_num_sweeps::Int

    MpsParameters(bd=typemax(Int), ϵ=1E-8, sw=4) = new(bd, ϵ, sw)
end


find_layout(network::PEPSNetwork{T}) where {T} = T 


struct MpsContractor <: AbstractContractor
    peps::PEPSNetwork{T} where T
    betas::Vector{Real}
    params::MpsParameters
    layers::MpoLayers

    function MpsContractor(peps, betas, params)
        T = find_layout(peps)
        layers = MpoLayers(T, peps.ncols)
        new(peps, betas, params, layers)
    end
end


function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EnergyGauges}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1:ncols push!(main, i => (-1//6, 0, 3//6, 4//6)) end
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end  

    dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

    for i ∈ 1:ncols push!(right, i => (-3//6, 0)) end
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end 

    MpoLayers(main, dress, right)
end


function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EnergyGauges}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1//2 : 1//2 : ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(main, ii => (-1//6, 0, 3//6, 4//6))
        push!(dress, ii => (3//6, 4//6))
        push!(right, ii => (-3//6, 0))
    end

    MpoLayers(main, dress, right)
end


function MpoLayers(::Type{T}, ncols::Int) where T <: Square{GaugesEnergy}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1:ncols push!(main, i => (-4//6, -1//2, 0, 1//6)) end
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end  

    dress = Dict(i => (1//6,) for i ∈ 1:ncols)

    for i ∈ 1:ncols push!(right, i => (-3//6, 0)) end
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end 

    MpoLayers(main, dress, right)
end


function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{GaugesEnergy}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1//2 : 1//2 : ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(main, ii => (-4//6, -1//2, 0, 1//6))
        push!(dress, ii => (1//6))
        push!(right, ii => (-3//6, 0))
    end

    MpoLayers(main, dress, right)
end


@memoize function mpo(contractor::MpsContractor, layers, r::Int, β::Real) where {T <: Number}
    W = Dict()
    # Threads.@threads for (j, coor) ∈ layers
    for (j, coor) ∈ layers
        push!(W,
            j => Dict(dr => tensor(contractor.peps, (r + dr, j), β) for dr ∈ coor)
        )
    end
    Mpo(W)
end


IdentityMps(peps::PEPSNetwork{T}) where T <: Square =
    Mps(Dict(j => ones(1, 1, 1) for j ∈ 1:peps.ncols))


function IdentityMps(peps::PEPSNetwork{T}) where T <: SquareStar
    id = Dict()
    for i ∈ 1//2 : 1//2 : peps.ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(id, ii => ones(1, 1, 1))
    end
    Mps(id)
end


@memoize function mps(contractor::MpsContractor, i::Int, β::Real) where {T <: Number}
    if i > contractor.peps.nrows return IdentityMps(contractor.peps) end  
    ψ = mps(contractor, i+1, β)
    W = mpo(contractor, contractor.layers.main, i, β)
    ψ0 = dot(W, ψ)   # dla rzadkosci nie mozemy tworzyc dot(W, ψ)
    # jako initial guess mozemy probowac wykorzystac mpsy z innych temperatur
    truncate!(ψ0, :left, contractor.params.bond_dimension)
    compress!(ψ0, W, ψ,
            contractor.params.bond_dimension,
            contractor.params.variational_tol,
            contractor.params.max_num_sweeps)
    ψ0
end


dressed_mps(contractor::MpsContractor, i::Int) = dressed_mps(contractor, i, last(contractor.betas))

@memoize Dict function dressed_mps(contractor::MpsContractor, i::Int, β::Real)
    ψ = mps(contractor, i+1, β)
    W = mpo(contractor, contractor.layers.dress, i, β)
    W * ψ
end


@memoize Dict function right_env(contractor::MpsContractor, i::Int, ∂v::Vector{Int}, β::Real)
    l = length(∂v)
    if l == 0 return ones(1, 1) end

    R̃ = right_env(contractor, i, ∂v[2:l], β)
    ϕ = dressed_mps(contractor, i, β)
    W = mpo(contractor, contractor.layers.right, i, β)
    k = length(ϕ.sites)
    site = ϕ.sites[k-l+1]
    M̃ = W[site]
    M = ϕ[site]

    RR = _update_reduced_env_right(R̃, ∂v[1], M̃, M)

    ls_mps = _left_nbrs_site(site, ϕ.sites)
    ls = _left_nbrs_site(site, W.sites)

    while ls > ls_mps
        M0 = W[ls][0]  ## struktura danych w mpo ???
        @tensor RR[x, y] := M0[y, z] * RR[x, z]
        ls = _left_nbrs_site(ls, W.sites)
    end
    RR
end


function _update_reduced_env_right(RE, m::Int, M::Dict, B) 
    M0 = M[0]
    Mt = M[-1//2]
    K = @view Mt[m, :]
    @tensor R[x, y] := K[d] * M0[y, d, β, γ] * 
                       B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
    R
end


@memoize Dict function left_env(contractor::MpsContractor, i::Int, ∂v::Vector{Int}, β::Real)
    l = length(∂v)
    if l == 0 return ones(1) end
    L̃ = left_env(contractor, i, ∂v[1:l-1], β)
    ϕ = dressed_mps(contractor, i, β)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]
    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    L
end


# function optimize_gauges(temp::MpsContractor)
#     #for beta in betas
    
#     # 1) psi_bottom =  mps  ;  psi_top = mps ( :top)
#     # 2) bazujac na psi_bottom i psi_top zmienia gauge
#     #    sweep left and right
        
#     #end
# end


function conditional_probability(contractor::MpsContractor, w::Vector{Int})
    T = find_layout(contractor.peps)
    β = last(contractor.betas)
    conditional_probability(T, contractor, w, β)
end


function conditional_probability(::Type{T}, contractor::MpsContractor, w::Vector{Int}, β) where T <: Square
    i, j = node_from_index(contractor.peps, length(w)+1)
    ∂v = boundary_state(contractor.peps, w, (i, j))

    L = left_env(contractor, i, ∂v[1:j-1], β)
    R = right_env(contractor, i, ∂v[j+2 : peps.ncols+1], β)
    ψ = dressed_mps(contractor, i, β)
    M = ψ.tensors[j]

    A = reduced_site_tensor(contractor.peps, (i, j), ∂v[j], ∂v[j+1], β)

    @tensor prob[σ] := L[x] * M[x, d, y] * A[r, d, σ] *
                       R[y, r] order = (x, d, r, y)

    normalize_probability(prob)
end


function conditional_probability(::Type{T}, contractor::MpsContractor, w::Vector{Int}, β) where T <: SquareStar
    i, j = node_from_index(contractor.peps, length(w)+1)
    ∂v = boundary_state(contractor.peps, w, (i, j))

    L = left_env(contractor, i, ∂v[1:2*j-2], β)
    R = right_env(contractor, i, ∂v[2*j+3 : 2*peps.ncols+2], β)
    ψ = dressed_mps(contractor, i, β)
    MX, M = ψ[j-1//2], ψ[j]

    A = reduced_site_tensor(contractor.peps, (i, j), ∂v[2*j-1], ∂v[2*j], ∂v[2*j+1], ∂v[2*j+2], β)

    @tensor prob[σ] := L[x] * MX[x, m, y] * M[y, l, z] * R[z, k] *
                        A[k, l, m, σ] order = (x, y, z, k, l, m)

    normalize_probability(prob)
end


function reduced_site_tensor(
    network::PEPSNetwork,
    v::Node,
    l::Int,
    u::Int,
    β::Real
)
    i, j = v
    eng_local = local_energy(network, v)
    pl = projector(network, v, (i, j-1))
    eng_pl = interaction_energy(network, v, (i, j-1))
    @matmul eng_left[x] := sum(y) pl[x, y] * eng_pl[y, $l]
    
    pu = projector(network, v, (i-1, j))
    eng_pu = interaction_energy(network, v, (i-1, j))
    @matmul eng_up[x] := sum(y) pu[x, y] * eng_pu[y, $u]

    en = eng_local .+ eng_left .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    pr = projector(network, v, (i, j+1))
    pd = projector(network, v, (i+1, j))
    @cast A[r, d, σ] := pr[σ, r] * pd[σ, d] * loc_exp[σ]
    A
end


function reduced_site_tensor(
    network::PEPSNetwork{T},
    v::Node,
    ld::Int,
    l::Int,
    d::Int,
    u::Int,
    β::Real
) where T <: SquareStar

    i, j = v
    eng_local = local_energy(network, v)

    pl = projector(network, v, (i, j-1))
    eng_pl = interaction_energy(network, v, (i, j-1))
    @matmul eng_left[x] := sum(y) pl[x, y] * eng_pl[y, $l]

    pd = projector(network, v, (i-1, j-1))
    eng_pd = interaction_energy(network, v, (i-1, j-1))
    @matmul eng_diag[x] := sum(y) pd[x, y] * eng_pd[y, $d]
    
    pu = projector(network, v, (i-1, j))
    eng_pu = interaction_energy(network, v, (i-1, j))
    @matmul eng_up[x] := sum(y) pu[x, y] * eng_pu[y, $u]

    en = eng_local .+ eng_left .+ eng_diag .+ eng_up
    loc_exp = exp.(-β .* (en .- minimum(en)))

    p_lb = projector(network, (i, j-1), (i+1, j))
    p_rb = projector(network, (i, j), (i+1, j-1))
    pr = projector(network, v, ((i+1, j+1), (i, j+1), (i-1, j+1)))
    pd = projector(network, v, (i+1, j))

    @cast A[r, d, (k̃, k), σ] := p_rb[σ, k] * p_lb[$ld, k̃] * pr[σ, r] * 
                                pd[σ, d] * loc_exp[σ]
    A
end
