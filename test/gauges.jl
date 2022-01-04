using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Memoize

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

β = 0.5
bond_dim = 16
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

fg = factor_graph(
    ising_graph(instance),
    max_cl_states,
    spectrum=brute_force,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)

Strategy = SVDTruncate
@testset "Updating gauges works correctly." begin
    for Layout ∈ (GaugesEnergy, ), transform ∈ rotation.([0])
        println((Strategy, Layout, transform))

        network_d = PEPSNetwork{Square{Layout}, Dense}(m, n, fg, transform, :rand)
        network_s = PEPSNetwork{Square{Layout}, Sparse}(m, n, fg, transform, :rand)
        ctr_d = MpsContractor{Strategy}(network_d, [β/8, β/4, β/2, β], params)
        ctr_s = MpsContractor{Strategy}(network_s, [β/8, β/4, β/2, β], params)

        @testset "Overlaps calculated differently agree" begin
            indβ = 3
            for i ∈ 1:n-1
                ψ_top = mps_top(ctr_d, i, indβ)
                ψ_bot = mps(ctr_d, i+1, indβ)
                overlap = tr(overlap_density_matrix(ψ_top, ψ_bot, indβ))
                @test overlap ≈ ψ_bot * ψ_top
            end
        end

        @testset "Test update_gauges" begin
            indβ = 3
            for i ∈ 1:n-1, i ∈ n-1:-1:1
                ψ_top = mps_top(ctr_d, i, indβ)
                ψ_bot = mps(ctr_d, i+1, indβ)
                overlap1 = tr(overlap_density_matrix(ψ_top, ψ_bot, indβ))
                update_gauges!(ctr_d, i, indβ)
                overlap2 = tr(overlap_density_matrix(ψ_top, ψ_bot, indβ))
                @test overlap1 ≈ overlap2
            end
        end
        clear_memoize_cache()
    end
end
