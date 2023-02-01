using SpinGlassExhaustive

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :CPU, num_states=num_states)
end

m = 6 # for Z3
n = 6
t = 4

β = 0.25
DE = 16.0
bond_dim = 5
δp = 1E-5*exp(-β * DE)
num_states = 128

ig = ising_graph("$(@__DIR__)/../instances/zephyr_random/Z3/RAU/SpinGlass/001_sg.txt")

fg = factor_graph(
    ig,
    # max_cl_states,
    spectrum = full_spectrum,  #brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule = zephyr_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = Zipper # SVDTruncate
Sparsity = Sparse #Dense
tran =  LatticeTransformation((4, 1, 2, 3), true)
Layout = GaugesEnergy
Gauge = NoUpdate

net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params)

# for i in 1//2 : 1//2 : m
#     for j in 1 : 1//2 : n
#         println("Size", (i,j)," = ", log2.(size(net, PEPSNode(i, j))))
#     end
# end

sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
println(sol.energies)

clear_memoize_cache()
