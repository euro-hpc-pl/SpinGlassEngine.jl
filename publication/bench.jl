using SpinGlassEngine
using SpinGlassNetworks

function get_instance(topology::NTuple{3, Int})
    m, n, t = topology
    "$(@__DIR__)/instances/square_diagonal/$(m)x$(n)x$(t).txt"
end

function run_square_diag_bench(::Type{T}; topology::NTuple{3, Int}) where {T}
    m, n, _ = topology
    instance = get_instance(topology)
    lattice = super_square_lattice(topology)

    best_energies = T[]

    potts_h = potts_hamiltonian(
        ising_graph(instance),
        spectrum = full_spectrum,
        cluster_assignment_rule = lattice,
    )

    params = MpsParameters{T}(; bond_dim = 16, num_sweeps = 1)
    search_params = SearchParameters(; max_states = 2^8, cutoff_prob = 1E-4)

    for transform ∈ all_lattice_transformations
        net = PEPSNetwork{KingSingleNode{GaugesEnergy}, Dense, T}(
            m, n, potts_h, transform,
        )

        ctr = MpsContractor{SVDTruncate, NoUpdate, T}(
            net, params; 
            onGPU = false, beta = T(2), graduate_truncation = true,
        )

        droplets = SingleLayerDroplets(; max_energy = 10, min_size = 5, metric = :hamming)
        merge_strategy = merge_branches(
            ctr; merge_prob = :none , droplets_encoding = droplets,
        )

        sol, info = low_energy_spectrum(ctr, search_params, merge_strategy)

        push!(best_energies, sol.energies[1])
        clear_memoize_cache()
    end

    ground = minimum(best_energies)
    @assert all(ground .≈ best_energies)

    println("Best energy found: $(ground)")
end

T = Float32
@time run_square_diag_bench(T; topology = (3, 3, 2))