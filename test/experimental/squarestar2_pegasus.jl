using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m = 7
n = 7
t = 3

β = 0.5
bond_dim = 8
DE = 16.0
δp = 1E-5*exp(-β * DE)
num_states = 128

VAR_TOL = 1E-16
MS = 0
TOL_SVD = 1E-16
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2
iter = 1
cs = 2^10

#ig = ising_graph("$(@__DIR__)/../instances/pegasus_droplets/4_4_3_00.txt")
ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P8/CBFM-P/SpinGlass/001_sg.txt")

cl_h = clustered_hamiltonian(
    ig,
    spectrum=my_brute_force, #rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)
# new_cl_h = clustered_hamiltonian_2site(cl_h, β)
# beliefs = belief_propagation(new_cl_h, β; tol=1e-6, iter=iter)
# cl_h = truncate_clustered_hamiltonian_2site_BP(cl_h, beliefs, cs; beta=β)

params = MpsParameters(bond_dim, VAR_TOL, MS, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = Zipper # MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
Layout = GaugesEnergy
Gauge = NoUpdate


for tran ∈ all_lattice_transformations #[LatticeTransformation((4, 3, 2, 1), false), ]
    println(" ==================== ")
    println(tran)

    net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, cl_h, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)

    sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    println(sol.energies)
    println(s)

clear_memoize_cache()

end