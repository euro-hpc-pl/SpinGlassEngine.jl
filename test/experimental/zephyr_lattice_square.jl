using SpinGlassEngine
using SpinGlassNetworks

m = 1
n = 1
t = 4

max_cl_states = 2^2

β = 3
bond_dim = 8
δp = 1e-10
num_states = 10

ig = ising_graph("$(@__DIR__)/../instances/zephyr/z1.txt")

fg = factor_graph(
    ig,
    # max_cl_states,
    spectrum=full_spectrum, #_gpu, # rm _gpu to use CPU
    cluster_assignment_rule=zephyr_lattice_5tuple((m,n,t))
)

clear_memoize_cache()