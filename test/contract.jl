using LightGraphs
using LabelledGraphs
using MetaGraphs


@testset "peps_contract correctly collapse the peps network" begin

    #      Grid
    #     A1    |    A2
    #           |
    #   1 -- 2 -|- 3

    D = Dict((1, 2) => -0.9049,
             (2, 3) =>  0.2838,

             (3, 3) => -0.7928,
             (2, 2) =>  0.1208,
             (1, 1) => -0.3342
    )

    m, n = 1, 2
    L = 4
    β = 1.

    ig = ising_graph(D)

    fg = factor_graph(
        ig,
        Dict((1, 1) => 4, (1, 2) => 2),
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 1), 3 => (1, 2), 4 => (1, 2))
    )

    e, p = get_prop(fg, (1, 1), (1, 2), :en), get_prop(fg, (1, 1), (1, 2), :pr)
    ϕ = exp(β * minimum(e * p))

    for i ∈ 1:4, j ∈ 1:2
        cfg = Dict((1, 1) => i, (1, 2) => j)

        Z = []
        for transform ∈ [rotation.([0, 90, 180, 270])..., reflection.([:x, :y, :diag, :antydiag])...]
            peps = PEPSNetwork(m, n, fg, transform)
            p = contract_network(peps, β, cfg)
            push!(Z, p)
        end

        # they all should be the same
        @test all(x -> x ≈ first(Z), Z)

        # the exact Gibbs state
        states = collect.(all_states(rank_vec(ig)))
        ρ = exp.(-β .* energy.(states, Ref(ig)))
        ϱ = reshape(ρ, (4, 2)) * ϕ

        # probabilities should agree
        @test first(Z) ≈ ϱ[cfg[(1, 1)], cfg[(1, 2)]]
    end
end
