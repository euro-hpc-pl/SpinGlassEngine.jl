export Mps
export compress!, dot, canonise, norm

abstract type AbstractEnvironment end

mutable struct Mps
    ket::Dict
    sites_ket
    function Mps(ket::Dict)
        sites_ket = sort(collect(keys(ket)))
        #ket = sort(collect(ket), by = x->x[1])
        mps = new(ket, sites_ket)
        mps
    end
end

mutable struct Environment <: AbstractEnvironment
    bra::Dict  # to be optimized
    mpo::Dict
    ket::Dict
    transpose::Bool
    sites_bra
    sites_mpo
    env::Dict

    function Environment(
        bra::Dict,
        mpo::Dict,
        ket::Dict,
        transpose::Bool = false
    )
        sites_bra = sort(collect(keys(bra)))
        sites_mpo = sort(collect(keys(mpo)))
        sites_ket = sort(collect(keys(ket)))

        @assert sites_bra == sites_ket
        @assert issubset(sites_bra, sites_mpo)

        # mpo can have sites with indices that are not in bra and ket
        env = Dict(
                   (first(sites_bra), :left) => ones(1, 1, 1),
                   (last(sites_bra), :right) => ones(1, 1, 1)
            )
        environment = new(bra, mpo, ket, transpose, sites_bra, sites_mpo, env)
        for site ∈ environment.sites_bra update_env_left!(environment, site) end
        environment
    end
end


function compress!(
    bra::Dict,
    mpo::Dict,
    ket::Dict,
    Dcut::Int,
    tol::Number=1E-8,
    max_sweeps::Int=4
)
    env = Environment(bra, mpo, ket, true)
    overlap_before = measure_env(env, last(env.sites_bra))

    for sweep ∈ 1:max_sweeps
        # left sweep
        for site ∈ reverse(env.sites_bra)
            update_env_right!(env, site)
            A = project_ket_on_bra(env, site)
            @cast B[x, (y, z)] := A[x, y, z]
            #_, Q = rq(B, typemax(Int))
            _, _, V = svd(B, typemax(Int)) 
            Q = V'
            @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
            env.bra[site] = C
            clear_env_site!(env, site)
        end
        
        # right sweep
        for site ∈ env.sites_bra
            update_env_left!(env, site)
            A = project_ket_on_bra(env, site)
            @cast B[(x, y), z] := A[x, y, z]
            #Q, _ = qr(B, typemax(Int))
            Q, _, _ = svd(B, typemax(Int))
            @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
            env.bra[site] = C
            clear_env_site!(env, site)
        end
        overlap = measure_env(env, last(env.sites_bra))

        Δ = abs(overlap_before - abs(overlap))
        @info "Convergence" Δ

        if Δ < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return 
        else
            overlap_before = overlap
        end
    end
end

# maximum(filter(x -> x < site, sites))
function _neighbouring_site_to_left(site, sites)
    # largest x in sites: x < site
    ind = first(sites)
    for x ∈ sites
        if x >= site break end
        ind = x
    end
    ind  # what should it return if there  is nothigh to the left?
end


# minimum(filter(x -> x > site, sites))
function _neighbouring_site_to_right(site, sites)
    # smallest x in sites: x > site
    ind = last(sites)
    for x ∈ reverse(sites)
        if x <= site break end
        ind = x
    end
    ind  # what should it return if there  is nothigh to the right
end


function update_env_left!(env::Environment, site)
    if site <= first(env.sites_bra) return end

    lsite = _neighbouring_site_to_left(site, env.sites_bra)
    LE = env.env[(lsite, :left)]
    T = env.bra[lsite]
    M = env.mpo[lsite]
    B = env.ket[lsite]
    Lnew = update_env_left(LE, T, M, B)

    isite = _neighbouring_site_to_right(lsite, env.sites_mpo)
    while isite < site
        iM = env.mpo[isite]
        Lnew = update_env_left(Lnew, iM)
        isite = _neighbouring_site_to_right(isite, env.sites_mpo)
    end
    push!(env.env, (site, :left) => Lnew)
end


function update_env_right!(env::Environment, site)
    if site >= last(env.sites_bra) return end

    rsite = _neighbouring_site_to_right(site, env.sites_bra)
    RE = env.env[(rsite, :right)]
    T = env.bra[rsite]
    M = env.mpo[rsite]
    B = env.ket[rsite]
    Rnew = update_env_right(RE, T, M, B)

    isite = _neighbouring_site_to_left(rsite, env.sites_mpo)
    while isite > site
        iM = env.mpo[isite]
        Rnew = update_env_right(Rnew, iM)
        isite = _neighbouring_site_to_left(isite, env.sites_mpo)
    end
    push!(env.env, (site, :right) => Rnew)
end


function clear_env_site!(env::Environment, site)
    # delete!(env.env, (site, :right))
    # delete!(env.env, (site, :left))
end


function update_env_left(LE, T, M, B)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * T[ot, α, nt] * 
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    # for real there is no conjugate, otherwise conj(T)
    L
end


#=
function update_env_left(LE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * T[ot, α, nt] * 
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    # for real there is no conjugate, otherwise conj(T)
    L
end
=#

function update_env_left(LE, iM)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[nb, oc, nt] * iM[oc, nc]
    L
end


function update_env_right(RE, T, M, B)  # same tensory bez wymiernych indeksow
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end

#=
function update_env_right(RE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc,  β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end
=#

function update_env_right(RE, iM)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor R[nt, nc, nb] := iM[nc, oc] * RE[nt, oc, nb]
    R
end


function project_ket_on_bra(env::Environment, site)
    LE = env.env[(site, :left)]
    M = env.mpo[site]
    B = env.ket[site]
    RE = env.env[(site, :right)]
    project_ket_on_bra(LE, B, M, RE)
end


function project_ket_on_bra(LE, B, M, RE)
    @tensor T[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    T
end

#=
function project_ket_on_bra(LE, B, M, RE, :transposed)
    @tensor T[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    T
end
=#

function measure_env(env::Environment, site)
    LE = env.env[(site, :left)]
    T = env.bra[site]
    M = env.mpo[site]
    B = env.ket[site]
    RE = env.env[(site, :right)]
    LL = update_env_left(LE, T, M, B)
    @tensor scalar = LL[t, c, b] * RE[b, c, t]
    scalar
end


"""
Calculates the norm of an MPS \$\\ket{\\phi}\$
"""
LinearAlgebra.norm(ψ::Mps) = sqrt(abs(dot(ψ, ψ)))

function LinearAlgebra.dot(ψ::Mps, ϕ::Mps)
    T = promote_type(eltype(ψ.ket[1]), eltype(ϕ.ket[1]))
    C = ones(T, 1, 1)

    for i ∈ ψ.sites_ket
        A, B = ψ.ket[i], ϕ.ket[i]
        @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
    end
    tr(C)
end


function canonise(ψ::Mps, s::Symbol, Dcut::Int=typemax(Int))
    @assert s ∈ (:left, :right)
    if s == :right
        nrm = _right_sweep!(ψ, typemax(Int))
        _left_sweep!(ψ, Dcut)
    else
        nrm = _left_sweep!(ψ, typemax(Int))
        _right_sweep!(ψ, Dcut)
    end
    abs(nrm)
end


function _right_sweep!(ψ::Mps, Dcut::Int=typemax(Int))
    R = ones(eltype(ψ.ket[1]), 1, 1)

    for i ∈ ψ.sites_ket
        A = ψ.ket[i]
        # attach
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]

        # decompose
        #Q, R = qr(M̃, Dcut)
        Q, S, V = svd(M̃, Dcut)
        R = Diagonal(S) * V'

        # create new
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ.ket[i] = A
    end
    R[1] 
end


function _left_sweep!(ψ::Mps, Dcut::Int=typemax(Int))
    R = ones(eltype(ψ.ket[1]), 1, 1)

    for i ∈ length(ψ.sites_ket):-1:1
        B = ψ.ket[i]

        # attach    
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]

        # decompose
        #R, Q = rq(M̃, Dcut)
        U, Σ, V = svd(M̃, Dcut) 
        R = U * Diagonal(Σ)
        Q = V'

        # create new
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ.ket[i] = B
    end
    R[1]
end