
using SpinGlassEngine

# ground_energy = -23.301855

m, n, t = 2, 4, 3
L = n * m * t

β = 1.0
bond_dim = 16
num_states = 10

instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_mdd.txt"

ig = ising_graph(instance)

fg = factor_graph(
    ig,
    spectrum=full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)
params = MpsParameters(bond_dim, 1E-8, 4)
search_params = SearchParameters(num_states, 0.0)



#for Strategy ∈ (SVDTruncate, MPSAnnealing), Sparsity ∈ (Sparse, Dense)
Strategy = SVDTruncate
Sparsity = Dense
Layout = EnergyGauges # GaugesEnergy, EngGaugesEng
transform = all_lattice_transformations[1]
Lattice = SquareStar
Gauge = NoUpdate

net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform)
ctr = MpsContractor{Strategy, Gauge}(net, [β/2, β], true, params)
sol = low_energy_spectrum(ctr, search_params)

ig_states = decode_factor_graph_state.(Ref(fg), sol.states)

states = decode_state.(Ref(net), sol.states)

norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
exact_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
unequal = [!(x ≈ y) for (x, y) in zip(norm_prob, exp.(-β .* (sol.energies .- sol.energies[1])))]
inds = findall(unequal)
println(inds)
println(sol.states[inds])
println(norm_prob[inds]./exact_prob[inds])


#clear_memoize_cache()

