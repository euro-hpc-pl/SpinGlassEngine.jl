using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m = 6 # for Z3
n = 6
t = 4

β = 0.5
DE = 16.0
bond_dim = 5
δp = 1E-5*exp(-β * DE)
num_states = 128
iter = 1
cs = 2^10
ig = ising_graph("$(@__DIR__)/../instances/zephyr_random/Z3/RAU/SpinGlass/001_sg.txt")

cl_h = clustered_hamiltonian(
    ig,
    # max_cl_states,
    spectrum = full_spectrum,  #brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule = zephyr_lattice((m, n, t))
)
new_cl_h = clustered_hamiltonian_2site(cl_h, β)
beliefs = belief_propagation(new_cl_h, β; tol=1e-6, iter=iter)
cl_h = truncate_clustered_hamiltonian_2site_BP(cl_h, beliefs, cs; beta=β)

params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = Zipper # SVDTruncate
Sparsity = Sparse #Dense
tran =  LatticeTransformation((4, 1, 2, 3), true)
Layout = GaugesEnergy
Gauge = NoUpdate

net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, cl_h, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)

# for i in 1//2 : 1//2 : m
#     for j in 1 : 1//2 : n
#         println("Size", (i,j)," = ", log2.(size(net, PEPSNode(i, j))))
#     end
# end

sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
println(sol.energies)

clear_memoize_cache()
