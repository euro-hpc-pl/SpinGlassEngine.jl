<<<<<<< HEAD
export compress!

abstract type AbstractEnvironment end

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
=======
export truncate!, canonise!, compress

function compress(chi::AbstractMPO, ψ::AbstractMPS, init_guess::Abstract_MPS, Dcut::Int, tol::Number=1E-8, max_sweeps::Int=4)

    # Initial guess - truncated ψ
    ϕ = copy(ψ)
    truncate!(ϕ, :right, Dcut)

    # Create environment
    env = left_env(ϕ, ψ)

    # Variational compression
    overlap = 0
    overlap_before = 1

    @info "Compressing down to" Dcut

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!!(ϕ, env, ψ, Dcut)
        overlap = _right_sweep_var!!(ϕ, env, ψ, Dcut)

        diff = abs(overlap_before - abs(overlap))
        @info "Convergence" diff

        if diff < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return ϕ
>>>>>>> sparse-mpo
        else
            overlap_before = overlap
        end
    end
<<<<<<< HEAD
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
=======
    ϕ
end

function canonise!(ψ::AbstractMPS)
    canonise!(ψ, :right)
    canonise!(ψ, :left)
end

canonise!(ψ::AbstractMPS, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::AbstractMPS, ::Val{:right}) = _left_sweep_SVD!(ψ)
canonise!(ψ::AbstractMPS, ::Val{:left}) = _right_sweep_SVD!(ψ)

truncate!(ψ::AbstractMPS, s::Symbol, Dcut::Int) = truncate!(ψ, Val(s), Dcut)
truncate!(ψ::AbstractMPS, ::Val{:right}, Dcut::Int) = _left_sweep_SVD!(ψ, Dcut)
truncate!(ψ::AbstractMPS, ::Val{:left}, Dcut::Int) = _right_sweep_SVD!(ψ, Dcut)
#=
function _right_sweep_SVD!(ψ::AbstractMPS, Dcut::Int=typemax(Int))
    Σ = V = ones(eltype(ψ), 1, 1)
    for (i, A) ∈ enumerate(ψ)
        C = (Diagonal(Σ) ./ Σ[1]) * V'

        # attach
        @matmul M̃[(x, σ), y] := sum(α) C[x, α] * A[α, σ, y]

        # decompose
        U, Σ, V = svd(M̃, Dcut)

        # create new
        d = physical_dim(ψ, i)
        @cast A[x, σ, y] |= U[(x, σ), y] (σ ∈ 1:d)
        ψ[i] = A
    end
    ψ[end] *= tr(V)
end


function _left_sweep_SVD!(ψ::AbstractMPS, Dcut::Int=typemax(Int))
    Σ = U = ones(eltype(ψ), 1, 1)
    for i ∈ length(ψ):-1:1
        B = ψ[i]
        C = U * (Diagonal(Σ) ./ Σ[1])

        # attach    
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * C[α, y]

        # decompose
        U, Σ, V = svd(M̃, Dcut)

        # create new
        d = physical_dim(ψ, i)
        @cast B[x, σ, y] |= V'[x, (σ, y)] (σ ∈ 1:d)
        ψ[i] = B
    end
    ψ[1] *= tr(U)
end
=#

function _left_sweep_var!!(ϕ::AbstractMPS, env::Vector{<:AbstractMatrix}, ψ::AbstractMPS, Dcut::Int)
    S = eltype(ϕ)

    # overwrite the overlap
    env[end] = ones(S, 1, 1)

    for i ∈ length(ψ):-1:1
        L = env[i]
        R = env[i+1]

        # optimize site
        M = ψ[i]
        @tensor MM[x, σ, α] := L[x, β] * M[β, σ, α] 
        @matmul MM[x, (σ, y)] := sum(α) MM[x, σ, α] * R[α, y]

        Q = rq(MM, Dcut)

        d = physical_dim(ψ, i)
        @cast B[x, σ, y] |= Q[x, (σ, y)] (σ ∈ 1:d)

        # update ϕ and right environment
        ϕ[i] = B
        A = ψ[i]

        @tensor RR[x, y] := A[x, σ, α] * R[α, β] * conj(B)[y, σ, β] order = (β, α, σ)
        env[i] = RR
    end
end


function _right_sweep_var!!(ϕ::AbstractMPS, env::Vector{<:AbstractMatrix}, ψ::AbstractMPS, Dcut::Int)
    S = eltype(ϕ)

    # overwrite the overlap
    env[1] = ones(S, 1, 1)

    for (i, M) ∈ enumerate(ψ)
        L = env[i]
        R = env[i+1]

        # optimize site
        @tensor M̃[x, σ, α] := L[x, β] * M[β, σ, α]
        @matmul B[(x, σ), y] := sum(α) M̃[x, σ, α] * R[α, y]

        Q = qr(B, Dcut)

        d = physical_dim(ψ, i)
        @cast A[x, σ, y] |= Q[(x, σ), y] (σ ∈ 1:d)

        # update ϕ and left environment
        ϕ[i] = A
        B = ψ[i]

        @tensor LL[x, y] := conj(A[β, σ, x]) * L[β, α] * B[α, σ, y] order = (α, β, σ)
        env[i+1] = LL
    end
    real(env[end][1])
end


function _right_sweep_SVD(::Type{T}, A::AbstractArray, Dcut::Int=typemax(Int), args...) where {T <: AbstractMPS}
    rank = ndims(A)
    ψ = T(eltype(A), rank)
    V = reshape(copy(conj(A)), (length(A), 1))

    for i ∈ 1:rank
        d = size(A, i)

        # reshape
        VV = conj.(transpose(V))
        @cast M[(x, σ), y] |= VV[x, (σ, y)] (σ ∈ 1:d)

        # decompose
        U, Σ, V = svd(M, Dcut, args...)
        V *= Diagonal(Σ)

        # create MPS
        @cast B[x, σ, y] |= U[(x, σ), y] (σ ∈ 1:d)
        ψ[i] = B
    end
    ψ
end


function _left_sweep_SVD(::Type{T}, A::AbstractArray, Dcut::Int=typemax(Int), args...) where {T <: AbstractMPS}
    rank = ndims(A)
    ψ = T(eltype(A), rank)

    U = reshape(copy(A), (length(A), 1))

    for i ∈ rank:-1:1
        d = size(A, i)

        # reshape
        @cast M[x, (σ, y)] |= U[(x, σ), y] (σ ∈ 1:d)

        # decompose
        U, Σ, V = svd(M, Dcut, args...)
        U *= Diagonal(Σ)

        # create MPS
        VV = conj.(transpose(V))
        @cast B[x, σ, y] |= VV[x, (σ, y)] (σ ∈ 1:d)
        ψ[i] = B
    end
    ψ
>>>>>>> sparse-mpo
end
