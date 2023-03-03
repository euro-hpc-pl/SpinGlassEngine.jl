using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m, n, t = 2, 2, 1

β = 1.0
bond_dim = 2
num_states = 7 #22

ig = ising_graph("$(@__DIR__)/../instances/square_gauss/S4/001.txt")
fg = factor_graph(
    ig,
    spectrum=my_brute_force,
    cluster_assignment_rule=periodic_lattice((m, n, t))
)
params = MpsParameters(bond_dim, 1E-8, 4)
search_params = SearchParameters(num_states, 0.0)
Gauge = NoUpdate

energies = Vector{Float64}[]
Strategy = SVDTruncate
Sparsity = Sparse
Layout = EnergyGauges
Lattice = Square
transform = rotation(0)

net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform)
ctr = MpsContractor{Strategy, Gauge}(net, [β/8., β/4., β/2., β], :graduate_truncate, params; onGPU=onGPU)
sol = gibbs_sampling(ctr, search_params, merge_branches(ctr))

println(sol)
#println(sol.states)