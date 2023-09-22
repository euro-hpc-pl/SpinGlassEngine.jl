using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m,n,t = 13, 4, 1
β = 1.0
bond_dim = 32
dE = 3.0
δp = exp(-β * dE)
num_states = 100
all_betas = [β/8, β/4, β/2, β]

instance = 

cl_h = clustered_hamiltonian(
    ising_graph(instance),
    max_cl_states,
    spectrum=my_brute_force,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)
params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
search_params = SearchParameters(num_states, δp)
transform = rotation(0)
Gauge = GaugeStrategy
Layout = GaugesEnergy

net = PEPSNetwork{SquareSingleNode{Layout}, Sparsity}(m, n, cl_h, transform)
ctr = MpsContractor{Strategy, Gauge}(net, all_betas, :graduate_truncate, params; onGPU=onGPU)
sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit))