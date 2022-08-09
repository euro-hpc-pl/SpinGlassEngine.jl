using SpinGlassExhaustive

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end

m = 4
n = 4
t = 3

β = 2
bond_dim = 4
δp = 1e-10
num_states = 128

ig = ising_graph("$(@__DIR__)/../instances/pegasus_droplets/4_4_3_00.txt")

fg = factor_graph(
    ig,
    spectrum= brute_force_gpu, #rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 2)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
tran =  rotation(0)
Layout = EnergyGauges
Gauge = NoUpdate

net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β/4, β/2, β], :graduate_truncate, params)

for i in 1//2 : 1//2 : m
    for j in 1 : 1//2 : n
        println("Size", (i,j)," = ", log2.(size(net, PEPSNode(i, j))))
    end
end

#sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))


clear_memoize_cache()
