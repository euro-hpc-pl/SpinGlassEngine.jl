export compress

mutable struct Environment
    bra::Dict,  # this will be optimized
    mpo::Dict,
    ket::Dict,
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
        sites_bra = sort(keys(bra))
        sites_mpo = sort(keys(mpo))
        sites_ket = sort(keys(bra))
        @assert sites_bra == sites_ket
        @assert issubset(sites_bra, site_mpo)  
        # last line follows convention that mpo can have
        # central sites with indices that are not in bra and ket
        env = Dict(
                   (first(sites_bra), :left) => ones(1, 1, 1),
                   (last(sites_bra), :right) => ones(1, 1, 1)
            )
        new(bra, mpo, ket, transpose, sites_bra, sites_mpo, env)
    end
end


function compress(W::Dict, ψ::Dict, Dcut::Int, tol::Number=1E-8, max_sweeps::Int=4, ψ₀::Dict)

    # initial guess + its canonization
    # ψ₀ :rand, :svd, Dict(mps)
    ket = copy(ψ)

    env = Environment(bra=ψ, mpo=W, ket=ket, transpose=true)
    _initialize_left_env!(env)
    overlap_before = measure_env(env, last(env.sites_bra))

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env, Dcut)
        _right_sweep_var!(env, Dcut)
        overlap = measure_env(env, last(env.sites_bra))

        Δ = abs(overlap_before - abs(overlap))
        @info "Convergence" Δ

        if Δ < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return ϕ
        else
            overlap_before = overlap
        end
    end
    ϕ
end


_initialize_left_env!(env) = for site ∈ env.sites_bra update_env!(env, site, :left) end


# maximum(filter(x -> x < site, sites))
function _neighbouring_site_to_left(site, sites)
    # largest x in sites: x < site
    last = first(sites)
    for x ∈ sites
        if x >= sites break end
        last = x
    end
    last  # what should it return if there  is nothigh to the left?
end

# minimum(filter(x -> x > site, sites))
function _neighbouring_site_to_right(site, sites)
    # smallest x in sites: x > site
    last = last(sites)
    for x ∈ reversed(sites)
        if x <= sites break end
        last = x
    end
    last  # what should it return if there  is nothigh to the right
end



function update_env_left!(env, site)
    if site <= first(env.sites_bra) return end

    lsite = _neighbour_site_to_left(site, env.sites_bra)
    LE = env.env[(lsite, :left)]
    T = env.bra[lsite]
    M = env.mpo[lsite]
    B = env.ket[lsite]
    Lnew = update_left_env(LE, T, M, B)

    isite = _neighbour_site_to_right(lsite, env.sites_mpo)
    while isite < site
        iM = env.mpo[isite]
        Lnew = update_left_env(Lnew, iM)
        isite = _neighbour_site_to_right(isite, env.sites_mpo)
    end
    push!(env.env, (r, :left) => Lnew)
end


function update_right_env!(env, site)
    if site >= last(env.sites_bra) return end

    rsite = _neighbour_site_to_right(site, env.sites_bra)
    RE = env.env[(rsite, :right)]
    T = env.bra[rsite]
    M = env.mpo[rsite]
    B = env.ket[rsite]
    Rnew = update_right_env(RE, T, M, B)

    isite = _neighbour_site_to_left(rsite, env.sites_mpo)
    while isite > site
        iM = env.mpo[isite]
        Rnew = update_right_env(Rnew, iM)
        isite = _neighbour_site_to_left(isite, env.sites_mpo)
    end
    push!(env.env, (r, :right) => Rnew)
end


function clear_env_site!(env, site)
    delate!(env.env, (site, :right))
    delate!(env.env, (site, :left))
end



function update_left_env(LE, T, M, B)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * T[ot, α, nt] * 
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    # for real there is no conjugate, otherwise conj(T)
    L
end


function update_left_env(LE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * T[ot, α, nt] * 
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    # for real there is no conjugate, otherwise conj(T)
    L
end


function update_left_env(LE, iM)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor L[nb, nc, nt] := LE[nb, oc, nt] * M[oc, nc]
    L
end


function update_right_env(RE, T, M, B)  # same tensory bez wymiernych indeksow
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end


function update_right_env(RE, T, M, B, :transposed)  # same tensory bez wymiernych indeksow
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * T[nt, α, ot] * M[nc,  β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    # for real there is no conjugate, otherwise conj(T)
    R
end


function update_right_env(RE, iM)  # same tensory bez wymiernych indeksow;  multiple dispatch for M sparse
    @tensor R[nt, nc, nb] := M[nc, oc] * RE[nt, oc, nb]
    R
end


function project_ket_on_bra(env, site)
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


function project_ket_on_bra(LE, B, M, RE, :transposed)
    @tensor T[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    T
end


function measure_env(env, site)
    LE = env.env[(site, :left)]
    T = env.bra[site]
    M = env.mpo[site]
    B = env.ket[site]
    RE = env.env[(site, :right)]
    LL = update_left_env(LE, T, M, B)
    @tensor scalar = LL[t, c, b] * RE[b, c, t]
    scalar
end


function _left_sweep_var!(env)
    for r ∈ reversed(env.sites_bra)
        update_right_env!(env, site)
        T = project_ket_on_bra(env, site)
        d = size(T, 2)
        @cast T2[x, (y, z)] := T[x, y, z]
        Q = rq(T2, Dcut)
        @cast T3[x, σ, y] |= Q[x, (σ, y)] (σ ∈ 1:d)
        env.bra[site] = T3
        clear_env_site!(env, site)
    end
end


function _right_sweep_var!(env)
    for r ∈ env.sites_bra
        update_left_env!(env, site)
        T = project_ket_on_bra(env, site)
        d = size(T, 2)
        @cast T2[(x, y), z] := T[x, y, z]
        Q = qr(T2, Dcut)
        @cast T3[x, σ, y] |= Q[(x, σ), y] (σ ∈ 1:d)
        env.bra[site] = T3
        clear_env_site!(env, site)
    end
end


# function canonise!(ψ::AbstractMPS)
#     canonise!(ψ, :right)
#     canonise!(ψ, :left)
# end

# canonise!(ψ::AbstractMPS, s::Symbol) = canonise!(ψ, Val(s))
# canonise!(ψ::AbstractMPS, ::Val{:right}) = _left_sweep_SVD!(ψ)
# canonise!(ψ::AbstractMPS, ::Val{:left}) = _right_sweep_SVD!(ψ)

# function _right_sweep_SVD!(ψ::AbstractMPS, Dcut::Int=typemax(Int))
#     Σ = V = ones(eltype(ψ), 1, 1)
#     for (i, A) ∈ enumerate(ψ)
#         C = (Diagonal(Σ) ./ Σ[1]) * V'

#         # attach
#         @matmul M̃[(x, σ), y] := sum(α) C[x, α] * A[α, σ, y]

#         # decompose
#         U, Σ, V = svd(M̃, Dcut)

#         # create new
#         d = physical_dim(ψ, i)
#         @cast A[x, σ, y] |= U[(x, σ), y] (σ ∈ 1:d)
#         ψ[i] = A
#     end
#     ψ[end] *= tr(V)
# end


# function _left_sweep_SVD!(ψ::AbstractMPS, Dcut::Int=typemax(Int))
#     Σ = U = ones(eltype(ψ), 1, 1)
#     for i ∈ length(ψ):-1:1
#         B = ψ[i]
#         C = U * (Diagonal(Σ) ./ Σ[1])

#         # attach    
#         @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * C[α, y]

#         # decompose
#         U, Σ, V = svd(M̃, Dcut)

#         # create new
#         d = physical_dim(ψ, i)
#         @cast B[x, σ, y] |= V'[x, (σ, y)] (σ ∈ 1:d)
#         ψ[i] = B
#     end
#     ψ[1] *= tr(U)
# end
