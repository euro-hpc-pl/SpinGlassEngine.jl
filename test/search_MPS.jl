
@testset "MPS based search finds the correct low energy spectrum" begin

    #instance = "$(@__DIR__)/instances/basic/128_001.txt"
    instance = "$(@__DIR__)/instances/pathological/cross_3_4_dd.txt"

    ig = ising_graph(instance)
    ig = prune(ig) 

    expected_energies = [-23.301855, -23.221513, -23.002799, -22.922457]

    max_states = 100
    to_show = length(expected_energies)

    dβ = 1.0/8.0
    β = 1.
    
    Dcut = 16
    var_ϵ = 1E-8
    max_sweeps = 4

    @testset "without purifications" begin
        schedule = fill(dβ, Int(ceil(β/dβ)))
        ψ = MPS(ig, Dcut, var_ϵ, max_sweeps, schedule)
        states, lprob, _ = solve(ψ, max_states)
        @test energy.(states[1:to_show], Ref(ig)) ≈ expected_energies
    end
    
#=
    @testset "LES" begin
        sol = low_energy_spectrum(
            ig, Dcut, var_ϵ, max_sweeps, 
            dβ, β, :log, max_states
        )
        @test sol.energies[1:to_show] ≈ expected_energies
    end
=#
end 
