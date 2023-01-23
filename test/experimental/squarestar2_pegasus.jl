using SpinGlassExhaustive

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end

m = 3
n = 3
t = 3

β = 2
bond_dim = 3
δp = 1e-10
num_states = 16

#ig = ising_graph("$(@__DIR__)/../instances/pegasus_droplets/4_4_3_00.txt")
ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P4/RAU/SpinGlass/001_sg.txt")

fg = factor_graph(
    ig,
    spectrum= brute_force_gpu, #rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 2)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = Zipper # MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
Layout = EnergyGauges
Gauge = NoUpdate


for tran ∈ all_lattice_transformations
    println(" ==================== ")
    println(tran)

    net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β/4, β/2, β], :graduate_truncate, params)

    @time begin
    sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    end
    println(sol.energies)

clear_memoize_cache()

end