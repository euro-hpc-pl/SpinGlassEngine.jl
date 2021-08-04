# ρ needs to be ∈ the right canonical form
function solve(ψ::AbstractMPS, keep::Int)
    @assert keep > 0 "Number of states has to be > 0"
    T = eltype(ψ)

    keep_extra = keep
    pCut = prob = 0.
    k = 1

    if keep < prod(rank(ψ))
        keep_extra += 1
    end

    states = fill([], 1, k)
    left_env = _make_left_env(ψ, k)

    for (i, M) ∈ enumerate(ψ)
        _, d, b = size(M)

        pdo = zeros(T, k, d)
        LL = _make_LL(ψ, b, k, d)
        config = zeros(Int, i, k, d)

        for j ∈ 1:k
            L = left_env[:, :, j]

            for σ ∈ local_basis(d)
                m = idx(σ)
                LL[:, :, j, m] = M[:, m, :]' * (L * M[:, m, :])
                pdo[j, m] = tr(LL[:, :, j, m])
                config[:, j, m] = vcat(states[:, j]..., σ)
            end
        end

        perm = collect(1: k * d)
        k = min(k * d, keep_extra)

        if k >= keep_extra
            partialsortperm!(perm, vec(pdo), 1:k, rev=true)
            prob = vec(pdo)[perm]
            pCut < last(prob) ? pCut = last(prob) : ()
        end

        @cast A[α, β, (l, d)] |= LL[α, β, l, d]
        left_env = A[:, :, perm]

        @cast B[α, (l, d)] |= config[α, l, d]
        states = B[:, perm]
    end
    states[:, 1:keep], prob[1:keep], pCut
end

_make_LL(ψ::AbstractMPS, b::Int, k::Int, d::Int) = zeros(eltype(ψ), b, b, k, d)
_make_left_env(ψ::AbstractMPS, k::Int) = ones(eltype(ψ), 1, 1, k)



function SpinGlassTensors.MPS(ig::LabelledGraph, control::MPSControl, type::Symbol)
    L = nv(ig)
    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    dβ = control.dβ
    β = control.β
    @info "Set control parameters for MPS" Dcut tol max_sweeps

    @info "Preparing Hadamard state as MPS"
    ρ = HadamardMPS(values(get_prop(ig, :rank)))
    is_right = true
    @info "Sweeping through β and σ" dβ

    if type == :log
        k = ceil(log2(β/dβ))
        dβmax = β/(2^k)
        ρ = _apply_layer_of_gates(ig, ρ, control, dβmax)
        for j ∈ 1:k
            ρ = multiply_purifications(ρ, ρ, L)
            if bond_dimension(ρ) > Dcut
                @info "Compresing MPS" bond_dimension(ρ), Dcut
                ρ = SpinGlassTensors.compress(ρ, Dcut, tol, max_sweeps)
                is_right = true
            end
        end
        ρ
    elseif type == :lin
        k = β/dβ
        dβmax = β/k
        ρ = _apply_layer_of_gates(ig, ρ, control, dβmax)
        ρ0 = copy(ρ)
        for j ∈ 1:k
            ρ = multiply_purifications(ρ, ρ0, L)
            if bond_dimension(ρ) > Dcut
                @info "Compresing MPS" bond_dimension(ρ), Dcut
                ρ = SpinGlassTensors.compress(ρ, Dcut, tol, max_sweeps)
                is_right = true
            end
        end
    end
    ρ
end
