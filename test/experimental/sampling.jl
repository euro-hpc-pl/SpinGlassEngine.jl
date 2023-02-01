using SpinGlassExhaustive

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end

m, n, t = 4, 4, 1

β = 1.0
bond_dim = 2
num_states = 7 #22

ig = ising_graph("$(@__DIR__)/../instances/square_gauss/S4/001.txt")
fg = factor_graph(
    ig,
    spectrum=brute_force_gpu,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)
params = MpsParameters(bond_dim, 1E-8, 4)
search_params = SearchParameters(num_states, 0.0)
Gauge = NoUpdate

energies = Vector{Float64}[]
Strategy = Zipper
Sparsity = Sparse
Layout = EnergyGauges
Lattice = Square
transform = rotation(0)

net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform)
ctr = MpsContractor{Strategy, Gauge}(net, [β/8., β/4., β/2., β], :graduate_truncate, params)
sol = gibbs_sampling(ctr, search_params)

println(sol.energies)
println(sol.states)