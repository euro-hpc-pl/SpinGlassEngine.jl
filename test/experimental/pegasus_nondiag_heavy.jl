using SpinGlassExhaustive

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end

m = 2
n = 2
t = 3

β = 3
bond_dim = 8
δp = 1e-10
num_states = 10

ig = ising_graph("$(@__DIR__)/../instances/pegasus_nondiag/pegasus_nd_2x2x3.txt")
#ig = ising_graph("$(@__DIR__)/../instances/chimera_droplets/128power/001.txt")

fg = factor_graph(
    ig,
    spectrum=brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 1)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
tran =  rotation(0)
Layout = GaugesEnergy
Gauge = NoUpdate

net = PEPSNetwork{PegasusSquare, Sparsity}(m, n, fg, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)

sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

# ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
# @test sol.energies ≈ energy.(Ref(ig), ig_states)
# fg_states = decode_state.(Ref(net), sol.states)
# @test sol.energies ≈ energy.(Ref(fg), fg_states)

# # norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
# # exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
# # @test norm_prob ≈ exct_prob

# println(sol.energies)

clear_memoize_cache()
