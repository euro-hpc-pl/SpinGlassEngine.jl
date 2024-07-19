using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

using SpinGlassExhaustive

onGPU = true

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

function bench(instance::String, β::Real, bond_dim::Integer, num_states::Integer)
    m, n, t = 15, 15, 3



    dE = 3.0
    δp = exp(-β * dE)
    all_betas = [β / 8, β / 4, β / 2, β]

    cl_h = clustered_hamiltonian(
        ising_graph(instance),
        spectrum = my_brute_force,
        cluster_assignment_rule = pegasus_lattice((m, n, t)),
    )

    params = MpsParameters{Float64}(;bd=bond_dim, ϵ=1E-8, sw=4, 1E-16)
    search_params = SearchParameters(; max_states=num_states, cut_off_prob=δp)
    Strategy = Zipper
    Sparsity = Sparse
    Layout = GaugesEnergy
    transform = rotation(0)
    Gauge = NoUpdate
    net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity,Float64}(m, n, cl_h, transform)
    ctr = MpsContractor{Strategy,Gauge,Float64}(
        net,
        all_betas,
        :graduate_truncate,
        params;
        onGPU = onGPU,
    )

    sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

    sol
end

β = 0.5
bond_dim = 8
num_states = 5

instance = "$(@__DIR__)/instances/P16_CBFM-P.txt"

bench(instance, β, bond_dim, num_states)
