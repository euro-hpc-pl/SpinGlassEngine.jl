using LabelledGraphs

export fuse_projectors, build_tensor_with_fusing


function fuse_projectors(projectors)
    fused, energy = rank_reveal(hcat(projectors...), :PE)
    i₀ = 1
    transitions = []
    for proj ∈ projectors
        iₑ = i₀ + size(proj, 2) - 1
        push!(transitions, energy[:, i₀:iₑ])
        i₀ = iₑ + 1
    end
    fused, transitions
end


# This has to be unified with build_tensor
@memoize function build_tensor_with_fusing(network::AbstractGibbsNetwork{S, T}, v::S) where {S, T}
    loc_exp = exp.(-network.β .* local_energy(network, v))

    #(pl, pb, pr, pt, trl, trr)
    projs, trl, trr = projectors_with_fusing(network, v) # only difference in comparison to build_tensor
    dim = zeros(Int, length(projs))
    @cast A[_, i] := loc_exp[i]

    for (j, pv) ∈ enumerate(projs)
        @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
        dim[j] = size(pv, 2)
    end
    reshape(A, dim..., :), trl, trr 
end