
@testset "MPS based search finds the correct low energy spectrum" begin

    instance = "$(@__DIR__)/instances/pathological/cross_3_4_dd.txt"

    ig = ising_graph(instance)
    ig = prune(ig) 

    expected_energies = [-23.301855, -23.221513, -23.002799, -22.922457, 
                         -22.394327, -22.457749, -22.546913, -22.466571
                        ]
    max_states = 100
    to_show = length(expected_energies)

    dβ = 0.01
    β = 2.

    D = 32
    var_ϵ = 1E-8
    sweeps = 4

    control = MPSControl(D, var_ϵ, sweeps, β, dβ)
    ψ = MPS(ig, control)
    states, lprob, _ = solve(ψ, max_states)

    @test energy.(states[1:to_show], Ref(ig)) ≈ expected_energies
end 
