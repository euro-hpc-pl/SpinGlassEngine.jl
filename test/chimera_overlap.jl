using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

ground_energy = -3336.773383

β = 0.5
bond_dim = 16
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

@time fg = factor_graph(
    ising_graph(instance),
    max_cl_states,
    spectrum=brute_force,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)

for Strategy ∈ (SVDTruncate, )
    for Layout ∈ (GaugesEnergy, ), transform ∈ rotation.([0])
        println((Strategy, Layout, transform))

        @time network_d = PEPSNetwork{Square{Layout}, Dense}(m, n, fg, transform)
        @time network_s = PEPSNetwork{Square{Layout}, Sparse}(m, n, fg, transform)
        @time ctr_d = MpsContractor{Strategy}(network_d, [β/8, β/4, β/2, β], params)
        @time ctr_s = MpsContractor{Strategy}(network_s, [β/8, β/4, β/2, β], params)

        for i in 1:n
            psi_bottom_d = mps(ctr_d, i, 4)
            psi_bottom_s = mps(ctr_s, i, 4)
            @test psi_bottom_d * psi_bottom_s ≈ 1.0
            #psi_top_d = mps_top(ctr_d, i, 4)
            #psi_top_s = mps_top(ctr_s, i, 4)
            #@test psi_top_d * psi_top_s ≈ 1.0
        end
        
        clear_memoize_cache()
            
    end
end
