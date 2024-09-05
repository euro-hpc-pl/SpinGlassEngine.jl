# This file is the code connected to example 3.1 in the paper
using SpinGlassPEPS

instance = "$(@__DIR__)/droplet_discovery_in_spin_glasses.txt"

m, n, t = 5, 5, 4
    
β = 1
bond_dim = 16
num_states = 100
dE = 3.0
δp = exp(-β * dE)
all_betas = [β/8, β/4, β/2, β]

# The size of our instance is 5X5 unit cells with
# 4 spins in each cell
cl_h = clustered_hamiltonian(ising_graph(instance),
    cluster_assignment_rule=super_square_lattice((5, 5, 4))
)

params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
search_params = SearchParameters(num_states, δp)
Strategy = Zipper
Sparsity = Sparse
Layout = GaugesEnergy
transform = rotation(0)
Gauge = NoUpdate


Node = SquareCrossSingleNode{Layout}
tensor_network = PEPSNetwork{Node, Sparsity}(5, 5, cl_h, transform)