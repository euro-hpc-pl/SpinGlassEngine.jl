using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

ground_energy = -3336.773383

β = 3.0
bond_dim = 48
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

for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, Sparse)
    for Layout ∈ (EnergyGauges, ), transform ∈ rotation.([0])
        println((Strategy, Sparsity, Layout, transform))

        @time network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
        @time ctr = MpsContractor{Strategy}(network, [β/8, β/4, β/2, β], params)

        psi_top1 = mps_top(ctr, 1, 4)
        psi_bottom2 = mps(ctr, 2, 4)
        psi_top2 = mps_top(ctr, 2, 4)
        psi_bottom3 = mps(ctr, 3, 4)
        psi_top3 = mps_top(ctr, 3, 4)
        psi_bottom4 = mps(ctr, 4, 4)

        overlap12 = psi_bottom2 * psi_top1 
        println("overlap 1-2 ", overlap12)
        overlap23 = psi_bottom3 * psi_top2 
        println("overlap 2-3 ", overlap23)
        overlap34 = psi_bottom4 * psi_top3 
        println("overlap 3-4 ", overlap34)

        clear_memoize_cache()
            
    end
end
