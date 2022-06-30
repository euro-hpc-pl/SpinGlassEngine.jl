using SpinGlassEngine

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end

m = 2
n = 2
t = 3

max_cl_states = 2^2

β = 3
bond_dim = 8
δp = 1e-10
num_states = 10

ig = ising_graph("$(@__DIR__)/../instances/pegasus_droplets/2_2_3_00.txt")

fg = factor_graph(
    ig,
    # max_cl_states,
    spectrum=full_spectrum, #_gpu, # rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice_masoud((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
tran = rotation(0)
Layout = GaugesEnergy
Gauge = NoUpdate

net = PEPSNetwork{PegasusSquareDiag, Sparsity}(m, n, fg, tran)

ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)

#sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

clear_memoize_cache()
