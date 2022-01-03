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
@testset "Compare the results for GaugesEnergy with Python" begin
    for Layout ∈ (GaugesEnergy, ), transform ∈ rotation.([0])
        println((Strategy, Layout, transform))

        network_d = PEPSNetwork{Square{Layout}, Dense}(m, n, fg, transform, :rand)
        network_s = PEPSNetwork{Square{Layout}, Sparse}(m, n, fg, transform, :rand)
        ctr_d = MpsContractor{Strategy}(network_d, [β/8, β/4, β/2, β], params)
        ctr_s = MpsContractor{Strategy}(network_s, [β/8, β/4, β/2, β], params)

        @testset "Compare the results for Dense with Python" begin
            for i in 1:n-1
                psi_top = mps_top(ctr_d, i, 4)
                psi_bottom = mps(ctr_d, i+1, 4)
                overlap = psi_bottom * psi_top
                println("overlap ", overlap)
            end
        end
        initialize_gauges!(ctr_d.peps, :id)

        println("cache size before: ", length(memoize_cache(mps)))
        @testset "Compare the results for Dense with Python" begin
            for i in 1:n-1
                psi_top = mps_top(ctr_d, i, 4)
                psi_bottom = mps(ctr_d, i+1, 4)
                overlap = psi_bottom * psi_top
                println("overlap ", overlap)
            end
        end
        println("cache size after ", length(memoize_cache(mps)))

        clear_memoize_cache()

    end
end
