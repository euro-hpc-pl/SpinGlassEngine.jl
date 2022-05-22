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
        for Gauge ∈ (GaugeStrategy, GaugeStrategyWithBalancing)
            println((Strategy, Layout, transform))

            network_d = PEPSNetwork{Square{Layout}, Dense}(m, n, fg, transform, :id)
            network_s = PEPSNetwork{Square{Layout}, Sparse}(m, n, fg, transform, :id)
            ctr_d = MpsContractor{Strategy, Gauge}(network_d, [β/8, β/4, β/2, β], true, params)
            ctr_s = MpsContractor{Strategy, Gauge}(network_s, [β/8, β/4, β/2, β], true, params)

            @testset "Overlaps calculated differently agree" begin
                indβ = 3
                for i ∈ 1:m-1
                    ψ_top = mps_top(ctr_d, i, indβ)
                    ψ_bot = mps(ctr_d, i+1, indβ)
                    overlap = tr(overlap_density_matrix(ψ_top, ψ_bot, indβ))
                    @test overlap ≈ ψ_bot * ψ_top
                    println("overlap ", overlap)
                end
            end

            @testset "Test update_gauges" begin
                indβ = [4, ]
                overlap_python = [0.2637787707674837, 0.2501621729619047, 0.2951954406837012]

                for i ∈ vcat(1:m-1)#, m-1:-1:1)
                    ψ_top = mps_top(ctr_d, i, indβ[begin])
                    ψ_bot = mps(ctr_d, i+1, indβ[begin])
                    overlap1 = ψ_top * ψ_bot
                    #@test overlap1 ≈ overlap_python[i]
                    @test isapprox(overlap1, overlap_python[i], atol=1e-5)
                    println("overlap1 ", overlap1)
                    #println("#(cache mps) = ", length(memoize_cache(mps)))
                    #println("#(cache mps_top) = ", length(memoize_cache(mps_top)))

                    update_gauges!(ctr_d, i, indβ, Val(:left))

                    ψ_top = mps_top(ctr_d, i, indβ[begin])
                    ψ_bot = mps(ctr_d, i+1, indβ[begin])
                    overlap2 = ψ_top * ψ_bot
                    println(overlap1, ' ', overlap2)

                    #println("#(cache mps after) = ", length(memoize_cache(mps)))
                    #println("#(cache mps_top after) = ", length(memoize_cache(mps_top)))

                    clear_memoize_cache()
                end
            end
            clear_memoize_cache()
        end
     end
 end