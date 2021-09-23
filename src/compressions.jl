export 
    Mps, Mpo

abstract type AbstractEnvironment end


mutable struct Mps
    ket::Dict
    sites
    Mps(ket::Dict) = new(ket, sort(collect(keys(ket))))
end


mutable struct Mpo 
    op::Dict
    sites
    Mpo(op::Dict) = new(op, sort(collect(keys(op))))
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


function SpinGlassTensors.compress!(
    bra::Dict,
    mpo::Dict,
    ket::Dict,
    Dcut::Int,
    tol::Number=1E-8,
    max_sweeps::Int=4,
    args...
)
    env = Environment(bra, mpo, ket, true)
    overlap = Inf
    overlap_before = measure_env(env, last(env.sites_bra))

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env, args...)
        _right_sweep_var!(env, args...)

        overlap = measure_env(env, last(env.sites_bra))

        Δ = abs(overlap_before - abs(overlap))
        @info "Convergence" Δ

        if Δ < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return overlap
        else
            overlap_before = overlap
        end
    end
    overlap
end


function _left_sweep_var!(env::Environment, args...)
    for site ∈ reverse(env.sites_bra)
        update_env_right!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[x, (y, z)] := A[x, y, z]
        _, Q = rq_fact(B, args...)
        @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_site!(env, site) 
    end
end


function _right_sweep_var!(env::Environment, args...)
    for site ∈ env.sites_bra
        update_env_left!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B, args...)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_site!(env, site)
    end
end


function _neighbouring_site_to_left(site, sites)
    # largest x in sites: x < site
    ind = -Inf
    for x ∈ sites
        if x >= site break end
        ind = x
    end
    ind  
end


function _neighbouring_site_to_right(site, sites)
    # smallest x in sites: x > site
    ind = Inf
    for x ∈ reverse(sites)
        if x <= site break end
        ind = x
    end
    ind  
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


function update_env_left(LE::S, T::S, M::S, B::S) where S <: AbstractMatrix 
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * T[ot, α, nt] * 
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    # for real there is no conjugate, otherwise conj(T)
    L
end


function update_env_left(LE::S, T0::S, M::Dict, B0::S) where S <: AbstractMatrix 
    ver_sites = collect(sort(keys(M)))

    T = T0
    for i ∈ ver_sites
        if i == 0 break end
        A = M[i]
        @tensor T[l, x, r] := T[l, y, r] * A[y, x]
    end

    B = B0
    for i ∈ reverse(ver_sites)
        if i == 0 break end
        A = M[i]
        @tensor B[l, x, r] := B[l, y, r] * A[x, y]
    end
    update_env_left(LE, T, M[0], B)
end

#=
function update_env_left(LE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * T[ot, α, nt] * 
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    # for real there is no conjugate, otherwise conj(T)
    L
end
=#

function update_env_left(LE::S, iM::S) where S <: AbstractMatrix # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[nb, oc, nt] * iM[oc, nc]
    L
end


function update_env_right(RE::S, T::S, M::S, B::S)  where S <: AbstractMatrix # same tensory bez wymiernych indeksow
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end


function update_env_right(RE::S, T0::S, M::Dict, B0::S) where S <: AbstractMatrix # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    ver_sites = collect(sort(keys(M)))
 
    T = T0
    for i in ver_sites
        if i == 0 break end
        A = M[i]
        @tensor T[l, x, r] := T[l, y, r] * A[y, x]
    end

    B = B0
    for i in reverse(ver_sites)
        if i == 0 break end
        A = M[i]
        @tensor B[l, x, r] := B[l, y, r] * A[x, y]
    end
    update_env_right(RE, T, M[0], B)
end

#=
function update_env_right(RE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc,  β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end
=#

function update_env_right(RE::S, iM::S) where S <: AbstractMatrix # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
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


function project_ket_on_bra(LE::S, B::S, M::S, RE::S) where S <: AbstractMatrix
    @tensor T[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    T
end

function project_ket_on_bra(LE::S, B0::S, M::Dict, RE::S) where S <: AbstractMatrix
    ver_sites = collect(sort(keys(M)))

    B = B0
    for i in reverse(ver_sites)
        if i == 0 break end
        A = M[i]
        @tensor B[l, x, r] := B[l, y, r] * A[x, y]
    end

    T = project_ket_on_bra(LE, B, M[0], RE)
    for i in ver_sites
        if i == 0 break end
        A = M[i]
        @tensor T[l, x, r] := T[l, y, r] * A[x, y]
    end
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


#=
is_left_normalized(ψ::Mps) = all(
    I(size(A, 3)) ≈ @tensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ) for A ∈ values(ψ.ket)
)

is_right_normalized(ϕ::Mps) = all(
    I(size(B, 1)) ≈ @tensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ) for B in values(ϕ.ket)
)
=#





