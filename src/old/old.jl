function conditional_probability(peps::PegasusNetwork, v::Vector{Int})
    i, j = node_from_index(peps, length(v)+1)
    ∂v = generate_boundary_states(peps, v, (i, j))

    L = _left_env(peps, i, ∂v[1:2*j-2])
    R = _right_env(peps, i, ∂v[2*j+2:peps.ncols*2+1])

    A, _, _ = build_tensor(peps, (i, j))
    v = build_tensor(peps, (i-1, j), (i, j)) 
    
    ψ = MPS(peps, i+1)
    MX = ψ[2*j-1]
    X = W[2*j-1]
    M = ψ[2*j]

    l, d, u = ∂v[2*j-1:2*j+1]
    #vt = v[u, :]
    #@tensor Ã[l, r, d, σ] := A[l, x, r, d, σ] * vt[x]
    @cast Ã[l, r, d, σ] := sum(x) A[l, x, r, d, σ] * v[$u, x]

    Xt = X[l, d, :, :]

    @tensor prob[σ] := L[x] * Xt[k, y] * MX[x, y, z] * M[z, l, m] *
                        Ã[k, n, l, σ] * R[m, n] order = (x, y, z, k, l, m, n)

    _normalize_probability(prob)