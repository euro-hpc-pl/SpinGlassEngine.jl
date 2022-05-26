using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Memoization
using ProgressMeter

 m = 4
 n = 4
 t = 8

 L = n * m * t
 max_cl_states = 2^(t-0)

 β = 2.0
 bond_dim = 32
 dE = 3.0
 δp = exp(-β * dE)
 num_states = 500
 indβ = 4

 graduate_truncation = :graduate_truncate

 instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

 fg = factor_graph(
     ising_graph(instance),
     max_cl_states,
     spectrum=brute_force,
     cluster_assignment_rule=super_square_lattice((m, n, t))
 )

 params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
 search_params = SearchParameters(num_states, δp)

 Strategy = SVDTruncate
 @testset "Memoization works correctly." begin
    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
        for Gauge ∈ (GaugeStrategy, )
            for Layout ∈ (GaugesEnergy,), transform ∈ rotation.([0]) 
                net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], graduate_truncation, params)

                
                println(ctr.peps.nrows)
                    
                indβ_vector = [4,]
                tol = 1E-4
                max_sweeps = 10

                for indβ in indβ_vector
                    for row ∈ 1:m-1
                        clm = ctr.layers.main
                        ψ_top = mps_top(ctr, row, indβ)
                        ψ_bot = mps(ctr, row + 1, indβ)
                    
                        ψ_top = deepcopy(ψ_top)
                        ψ_bot = deepcopy(ψ_bot)
                    
                        gauges = optimize_gauges_for_overlaps!!(ψ_top, ψ_bot, tol, max_sweeps)
                        overlap = ψ_top * ψ_bot

                    
                        for i ∈ ψ_top.sites
                            g = gauges[i]
                            g_inv = 1.0 ./ g
                            @inbounds n_bot = PEPSNode(row + 1 + clm[i][begin], i)
                            @inbounds n_top = PEPSNode(row + clm[i][end], i)
                            g_top = ctr.peps.gauges.data[n_top] .* g
                            g_bot = ctr.peps.gauges.data[n_bot] .* g_inv
                            push!(ctr.peps.gauges.data, n_top => g_top, n_bot => g_bot)
                        end
                        
                        before_mps = length(Memoization.caches[mps])
                        before_mps_top = length(Memoization.caches[mps_top])
                        before_mpo =  length(Memoization.caches[SpinGlassEngine.mpo])

                        #weird error
                        #print(collect(values(Memoization.caches[mps_top]))) 

                        println("Before: mps, mps_top, mpo ", before_mps, " ", before_mps_top, " ", before_mpo)

                        clear_memoize_cache(ctr, row, indβ)

                        after_mps = length(Memoization.caches[mps])
                        after_mps_top = length(Memoization.caches[mps_top])
                        after_mpo =  length(Memoization.caches[SpinGlassEngine.mpo])

                        println("After: mps, mps_top, mpo ", after_mps, " ", after_mps_top, " ", after_mpo)
                    end
                end


               #= @testset "clear_memoize_cache deletes proper number of elements" begin


                end

                @testset "clear_memoize_cache deletes proper elements" begin
                    #∉
                    
                
                end=#
                


            end
        end
    end

end