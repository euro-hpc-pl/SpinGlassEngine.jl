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
    env::Dict

    function Environment(
        bra::Dict,
        mpo::Dict,
        ket::Dict,
        trans::Symbol=:c
    )
        @assert trans ∈ (:n, :c)
        @assert bra.sites == ket.sites
        @assert issubset(bra.sites, mpo.sites)

        # mpo can have sites with indices that are not in bra and ket
        env = Dict(
                   (first(bra.sites), :left) => ones(1, 1, 1),
                   (last(bra.sites), :right) => ones(1, 1, 1)
            )
        environment = new(bra, mpo, ket, trans, env)
        for site ∈ environment.bra.sites 
            update_env_left!(environment, site) 
        end
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
    env = Environment(bra, mpo, ket)
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
    for site ∈ reverse(env.bra.sites)
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
    for site ∈ env.bra.sites
        update_env_left!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B, args...)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_site!(env, site)
    end
end


function _left_nbrs_site(site, sites)
    # largest x in sites: x < site
    ls = filter(i -> i < site, sites)
    if isempty(ls) return -Inf end
    maximum(ls)
end


function _right_nbrs_site(site, sites)
    # smallest x in sites: x > site
    ms = filter(i -> i > site, sites)
    if isempty(ms) return Inf
    minimum(ms)
end


function update_env_left!(env::Environment, site)
    if site <= first(env.bra.sites) return end

    lsite = _left_nbrs_site(site, env.bra.sites)
    Lnew = update_env_left(
                env.env[(lsite, :left)],
                env.bra[lsite],
                env.mpo[lsite],
                env.ket[lsite],
    )

    isite = _right_nbrs_site(lsite, env.mpo.sites)

    while isite < site
        iM = env.mpo[isite]
        Lnew = update_env_left(Lnew, iM)
        isite = _right_nbrs_site(isite, env.mpo.sites)
    end
    push!(env.env, (site, :left) => Lnew)
end


function update_env_right!(env::Environment, site)
    if site >= last(env.sites_bra) return end

    rsite = _right_nbrs_site(site, env.bra.sites)
    Rnew = update_env_right(  
          RE = env.env[(rsite, :right)]
          T = env.bra[rsite]
          M = env.mpo[rsite]
          B = env.ket[rsite]
    )

    isite = _left_nbrs_site(rsite, env.mpo.sites)
    while isite > site
        iM = env.mpo[isite]
        Rnew = update_env_right(Rnew, iM)
        isite = _left_nbrs_site(isite, env.mpo.sites)
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


function _update_tensor_forward!(T::S, M::Dict, sites) where S <: AbstractMatrix 
    for i ∈ sites
        if i == 0 break end
        A = M[i]
        @tensor T[l, x, r] := T[l, y, r] * A[y, x]
    end
end


function _update_tensor_backwards!(T::S, M::Dict, sites) where S <: AbstractMatrix 
    for i ∈ reverse(sites)
        if i == 0 break end
        A = M[i]
        @tensor T[l, x, r] := T[l, y, r] * A[x, y]
    end
end


function update_env_left(LE::S, T₀::S, M::Dict, B₀::S) where S <: AbstractMatrix 
    sites = collect(sort(keys(M)))
    _update_tensor_forward!(T₀, M, sites)
    _update_tensor_backwards!(B₀, M, sites)
    update_env_left(LE, T₀, M[0], B₀)
end


#=
function update_env_left(LE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * T[ot, α, nt] * 
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    # for real there is no conjugate, otherwise conj(T)
    L
end
=#

function update_env_left(LE::S, iM::S) where S <: AbstractMatrix 
    L
end


function update_env_right(RE::S, T::S, M::S, B::S)  where S <: AbstractMatrix #
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end


function update_env_right(RE::S, T₀::S, M::Dict, B₀::S) where S <: AbstractMatrix 
    sites = collect(sort(keys(M)))
    _update_tensor_forward!(T₀, M, sites)
    _update_tensor_backwards!(B₀, M, sites)
    update_env_right(RE, T₀, M[0], B₀)
end

#=
function update_env_right(RE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc,  β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end
=#

function update_env_right(RE::S, iM::S) where S <: AbstractMatrix 
    @tensor R[nt, nc, nb] := iM[nc, oc] * RE[nt, oc, nb]
    R
end


function project_ket_on_bra(env::Environment, site)
    project_ket_on_bra(
        env.env[(site, :left)]
        env.ket[site]
        env.mpo[site]
        env.env[(site, :right)]
    )
end


function project_ket_on_bra(LE::S, B::S, M::S, RE::S) where S <: AbstractMatrix
    @tensor T[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    T
end


function project_ket_on_bra(LE::S, B₀::S, M::Dict, RE::S) where S <: AbstractMatrix
    sites = collect(sort(keys(M)))
    _update_tensor_forward!(B₀, M, sites)
    T = project_ket_on_bra(LE, B₀, M[0], RE)
    _update_tensor_backwards(T, M, sites)
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
    LL = update_env_left(
        env.env[(site, :left)]
        env.bra[site]
        env.mpo[site]
        env.ket[site]
    )
    LE = env.env[(site, :left)]
    @tensor LL[t, c, b] * RE[b, c, t]
end


