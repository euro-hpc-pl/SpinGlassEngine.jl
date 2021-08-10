using LabelledGraphs

export 
    solve,
    low_energy_spectrum

_make_left_env(ψ::AbstractMPS, k::Int) = ones(eltype(ψ), 1, k)

_make_LL(ψ::AbstractMPS, b::Int, k::Int, d::Int) = zeros(eltype(ψ), b, k, d)


# to be removed
function prune(ig::IsingGraph) 
    idx = findall(!iszero, degree(ig))
    gg = ig[ig.labels[idx]]
    labels = collect(vertices(gg.inner_graph))
    reverse_label_map = Dict(i => i for i=1:nv(gg.inner_graph))
    LabelledGraph(labels, gg.inner_graph, reverse_label_map)
end


function low_energy_spectrum(
    ig::IsingGraph,
    Dcut::Int, 
    var_ϵ::T, 
    max_sweeps::Int, 
    dβ::T,
    β::T,
    schedule::Symbol, 
    num_states::Int
) where T <: Number

    igp = prune(ig) 
    ψ = MPS(igp, Dcut, var_ϵ, max_sweeps, dβ, β, schedule)
    states, probs, ldp = solve(ψ, num_states)

    Solution(
        energy.(states, Ref(igp)),
        states,
        probs,
        [0],
        ldp
    )
end 

function solve(ψ::AbstractMPS, keep::Int)
    @assert keep > 0 "Number of states has to be > 0"
    T = eltype(ψ)

    keep_extra = keep
    lpCut = -Inf
    k = 1

    if keep < prod(rank(ψ)) keep_extra += 1 end

    lprob = zeros(T, k)
    states = fill([], 1, k)
    left_env = _make_left_env(ψ, k)

    for (i, M) ∈ enumerate(ψ)
        _, d, b = size(M)

        pdo = ones(T, k, d)
        lpdo = zeros(T, k, d)
        LL = _make_LL(ψ, b, k, d)
        config = zeros(Int, i, k, d)

        for j ∈ 1:k
            L = left_env[:, j]

            for σ ∈ local_basis(d)
                m = idx(σ)
                LL[:, j, m] = L' * M[:, m, :]
                pdo[j, m] = dot(LL[:, j, m], LL[:, j, m])
                config[:, j, m] = vcat(states[:, j]..., σ)
                LL[:, j, m] = LL[:, j, m] / sqrt(pdo[j, m])
            end
            pdo[j, :] = pdo[j, :] / sum(pdo[j, :])
            lpdo[j, :] = log.(pdo[j, :]) .+ lprob[j]
        end

        perm = collect(1 : k * d)
        k = k * d

        if k > keep_extra
            k = keep_extra
            partialsortperm!(perm, vec(lpdo), 1:k, rev=true)
            lprob = vec(lpdo)[perm]
            lpCut < last(lprob) ? lpCut = last(lprob) : ()
        end

        lprob = vec(lpdo)[perm]
        @cast A[α, (l, d)] |= LL[α, l, d]
        left_env = A[:, perm]
        @cast B[β, (l, d)] |= config[β, l, d]
        states = B[:, perm]
    end
    Vector.(eachcol(states[:, 1:keep])), lprob[1:keep], lpCut
end


function _apply_bias!(
    ψ::AbstractMPS,
    ig::LabelledGraph, 
    dβ::Number, 
    i::Int
)
    M = ψ[i]
    h = get_prop(ig, i, :h)
    σ = local_basis(ψ, i)
    v = exp.(-0.5 * dβ * h * σ)
    @cast M[x, σ, y] = M[x, σ, y] * v[σ]
    ψ[i] = M
end


function _apply_exponent!(
    ψ::AbstractMPS, 
    ig::LabelledGraph, 
    dβ::Number, 
    i::Int, 
    j::Int, 
    last::Int
)
    M = ψ[j]
    D = typeof(M).name.wrapper(I(physical_dim(ψ, i)))

    J = get_prop(ig, i, j, :J)
    σ = local_basis(ψ, i)
    η = local_basis(ψ, j)'
    C = exp.(-0.5 * dβ * σ *J * η )

    if j == last
        @cast M̃[(x, a), σ, b] := C[x, σ] * M[a, σ, b]
    else
        @cast M̃[(x, a), σ, (y, b)] := C[x, σ] * D[x, y] * M[a, σ, b]
    end
    ψ[j] = M̃
end


function _apply_projector!(ψ::AbstractMPS, i::Int)
    M = ψ[i]
    D = typeof(M).name.wrapper(I(physical_dim(ψ, i)))
    @cast M̃[a, σ, (y, b)] := D[σ, y] * M[a, σ, b]
    ψ[i] = M̃
end


function _apply_nothing!(ψ::AbstractMPS, l::Int, i::Int)
    M = ψ[l]
    D = typeof(M).name.wrapper(I(physical_dim(ψ, i)))
    @cast M̃[(x, a), σ, (y, b)] := D[x, y] * M[a, σ, b]
    ψ[l] = M̃
end


function purifications(χ::T, ϕ::T) where {T <: AbstractMPS}
    S = promote_type(eltype(χ), eltype(ϕ))
    ψ = MPS(S, length(ϕ))
    for (i, (A, B)) ∈ enumerate(zip(χ, ϕ))
        @cast C[(l, l̃), σ, (r, r̃)] := A[l, σ, r] * B[l̃, σ, r̃]
        ψ[i] = C
    end
    ψ
end
purifications(χ) = purifications(χ, χ) 

_holes(l::Int, nbrs::Vector) = setdiff(l+1:last(nbrs), nbrs)

function ___svd(A::AbstractMatrix, Dcut::Int, args...)
    U, Σ, V = psvd(A, rank=Dcut, args...)
    d = U[1, :]
    d[d .≈ 0] .= -1
    ph = d ./ abs.(d)
    return  U * Diagonal(ph), Σ, V * Diagonal(ph)
end

function _apply_gates(
    ρ::AbstractMPS, 
    ig::IsingGraph, 
    Dcut::Int,
    var_ϵ::Number,
    sweeps::Int,
    dβ::Number
)
    is_right = true
    for i ∈ 1:nv(ig)
        _apply_bias!(ρ, ig, dβ, i)
        is_right = false

        nbrs = unique_neighbors(ig, i)
        if !isempty(nbrs)
            _apply_projector!(ρ, i)
            for j ∈ nbrs
                _apply_exponent!(ρ, ig, dβ, i, j, last(nbrs))
            end
            for l ∈ _holes(i, nbrs)
                _apply_nothing!(ρ, l, i)
            end
        end

        if bond_dimension(ρ) > Dcut
            ρ = SpinGlassTensors.compress(ρ, Dcut, var_ϵ, sweeps)
            is_right = true
        end
    end

    if !is_right SpinGlassTensors.canonise!(ρ, :right) end
    ρ
end


function SpinGlassTensors.MPS(
    ig::IsingGraph, 
    Dcut::Int,
    var_ϵ::Number,
    sweeps::Int,
    schedule::Vector{<:Number}
)
    ρ = HadamardMPS(ig)
    for dβ ∈ schedule
        ρ =_apply_gates(ρ, ig, Dcut, var_ϵ, sweeps, dβ)
    end
    ρ
end


function SpinGlassTensors.MPS(
    ig::IsingGraph,
    Dcut::Int,
    var_ϵ::Number,
    sweeps::Int,
    dβ::Number,
    β::Number,
    schedule::Symbol
)
    ρ = HadamardMPS(ig)
    is_right = true

    if schedule == :log
        k = Int(ceil(log2(β/dβ)))
        ρ =_apply_gates(ρ, ig, Dcut, var_ϵ, sweeps, β/(2^k))
        for _ ∈ 1:k
            ρ = purifications(ρ)
            if bond_dimension(ρ) > Dcut
                ρ = SpinGlassTensors.compress(ρ, Dcut, var_ϵ, sweeps)
                is_right = true
            end
        end
    elseif schedule == :lin
        k = ceil(β/dβ)
        ρ = _apply_gates(ρ, ig, Dcut, var_ϵ, sweeps, β/k)
        ρ0 = copy(ρ)
        for _ ∈ 1:Int(k)
            ρ = purifications(ρ, ρ0)
            if bond_dimension(ρ) > Dcut
                ρ = SpinGlassTensors.compress(ρ, Dcut, var_ϵ, sweeps)
                is_right = true
            end
        end
    end
    if !is_right SpinGlassTensors.canonise!(ρ, :right) end
    ρ
end


function HadamardMPS(::Type{T}, ig::IsingGraph) where T <: Number
    MPS([
        fill(one(T), r) ./ sqrt(T(r)) 
        for r ∈ values(get_prop(ig, :rank))
    ])
end
HadamardMPS(ig) = HadamardMPS(Float64, ig)
