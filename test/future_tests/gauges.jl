using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Memoize

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

β = 2
bond_dim = 32
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

@testset "Updating gauges works correctly." begin
for Strategy ∈ (SVDTruncate, MPSAnnealing), Sparsity ∈ (Dense, Sparse) 
    for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
        for Gauge ∈ (GaugeStrategy, )
            for Lattice ∈ (Square, SquareStar), transform ∈ all_lattice_transformations
                net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform, :id)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)

                @testset "Overlaps calculated differently are the same." begin
                    indβ = 3
                    for i ∈ 1:m-1
                        ψ_top = mps_top(ctr, i, indβ)
                        ψ_bot = mps(ctr, i+1, indβ)
                        overlap = tr(overlap_density_matrix(ψ_top, ψ_bot, indβ))
                        @test overlap ≈ ψ_bot * ψ_top
                    end
                end
                clear_memoize_cache()

                @testset "Gauges are correctly optimized and updated." begin
                    indβ = [3,]
                    for _ in 1:3, i ∈ 1:m-1
                        ψ_top = mps_top(ctr, i, indβ[begin])
                        ψ_bot = mps(ctr, i+1, indβ[begin])

                        overlap_old = ψ_top * ψ_bot

                        #overlap_new = update_gauges!(ctr, i, indβ, Val(:left))
                        update_gauges!(ctr, i, indβ, Val(:down))

                        # assert that ψ_bot and ψ_top are not updated in place though memoize!
                        overlap_old2 = ψ_bot * ψ_top

                        # should be calculated from scratch with updated gauges
                        #ψ_top2 = mps_top(ctr, i, indβ[begin])
                        #ψ_bot2 = mps(ctr, i+1, indβ[begin])
                        
                        #overlap_new = ψ_top2 * ψ_bot2

                        @test overlap_old ≈ overlap_old2
                        #@test abs((overlap_new - overlap_new2) / overlap_new) < 1e-4
                        #@test overlap_new > overlap_old
                    end
                    update_gauges!(ctr, m, indβ, Val(:down))
                    sol_l = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
                    clear_memoize_cache()
                    update_gauges!(ctr, m, indβ, Val(:up))
                    sol_r = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
                    @test sol_l.energies[begin] ≈ sol_r.energies[begin]
                end
                clear_memoize_cache()
            end
        end
    end
end
end