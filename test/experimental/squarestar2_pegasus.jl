using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m = 3
n = 3
t = 3

β = 0.5
bond_dim = 8
DE = 16.0
δp = 1E-5*exp(-β * DE)
num_states = 128

VAR_TOL = 1E-16
MS = 0
TOL_SVD = 1E-16
ITERS_SVD = 0
ITERS_VAR = 0

#ig = ising_graph("$(@__DIR__)/../instances/pegasus_droplets/4_4_3_00.txt")
ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P4/CBFM-P/SpinGlass/single/001_sg.txt")

fg = factor_graph(
    ig,
    spectrum=my_brute_force, #rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

params = MpsParameters(bond_dim, VAR_TOL, MS, TOL_SVD, ITERS_SVD, ITERS_VAR)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = Zipper # MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
Layout = GaugesEnergy
Gauge = NoUpdate


for tran ∈ [LatticeTransformation((4, 3, 2, 1), false), ]
    println(" ==================== ")
    println(tran)

    net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)

    sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    println(sol.energies)
    println(s)

clear_memoize_cache()

end