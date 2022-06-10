using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Memoize

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

β =0.5
bond_dim = 32
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/../instances/chimera_droplets/128power/001.txt"

fg = factor_graph(
    ising_graph(instance),
    max_cl_states,
    spectrum=brute_force,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)

@testset "Overlaps calculated differently are the same." begin
for Lattice ∈ (Square, SquareStar) 
    for Sparsity ∈ (Dense, Sparse), transform ∈ all_lattice_transformations[[1]]
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform, :id)
            ctr_svd = MpsContractor{SVDTruncate, GaugeStrategy}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)
            ctr_anneal = MpsContractor{MPSAnnealing, GaugeStrategy}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)

            @testset "Overlaps calculated for different Starategies are the same." begin
                indβ = 3
                for i ∈ 1:m-1
                    ψ_top = mps_top(ctr_svd, i, indβ)
                    ϕ_top = mps_top(ctr_anneal, i, indβ)
                    @test ψ_top * ψ_top ≈ 1
                    @test ϕ_top * ϕ_top ≈ 1
                    @test ψ_top * ϕ_top ≈ 1
                end
            end
        end
        clear_memoize_cache()

        for Layout ∈ (GaugesEnergy, )
            net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform, :id)
            ctr_svd = MpsContractor{SVDTruncate, GaugeStrategy}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)
            @testset "Overlaps calculated in Python are the same as in Julia." begin
                indβ = [4, ]
                overlap_python = [0.2637787707674837, 0.2501621729619047, 0.2951954406837012]

                for i ∈ vcat(1:m-1)#, m-1:-1:1)
                    ψ_top = mps_top(ctr_svd, i, indβ[begin])
                    ψ_bot = mps(ctr_svd, i+1, indβ[begin])
                    overlap1 = ψ_top * ψ_bot
                    @test isapprox(overlap1, overlap_python[i], atol=1e-5)
                end
                clear_memoize_cache()
            end
        end
    end
end
end

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

                @testset "ψ_bot and ψ_top are not updated in place though memoize!" begin
                    indβ = [3,]
                    for _ in 1:3, i ∈ 1:m-1
                        ψ_top = mps_top(ctr, i, indβ[begin])
                        ψ_bot = mps(ctr, i+1, indβ[begin])

                        overlap_old = ψ_top * ψ_bot

                        update_gauges!(ctr, i, indβ, Val(:down))

                        # assert that ψ_bot and ψ_top are not updated in place though memoize!
                        overlap_old2 = ψ_bot * ψ_top

                        @test overlap_old ≈ overlap_old2

                    end
                end
                clear_memoize_cache()

                @testset "Updating gauges from top and bottom gives the same energy." begin
                    indβ = [3,]
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