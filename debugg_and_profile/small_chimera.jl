using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Logging
using Profile, PProf

disable_logging(LogLevel(1))

function bench(instance::String)
    m = 3
    n = 4
    t = 3
    L = n * m * t

    ground_energy = -16.4

    β = 3.0
    bond_dim = 16
    δp = 1E-3
    num_states = 1000

    cl_h = clustered_hamiltonian(
        ising_graph(instance),
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    net = PEPSNetwork{SquareSingleNode{EnergyGauges}, Sparse}(m, n, cl_h, rotation(0))
    ctr = MpsContractor{SVDTruncate}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)
    sol, s = low_energy_spectrum(ctr, search_params)#, merge_branches(ctr))

    @assert sol.energies[begin] ≈ ground_energy
    clear_memoize_cache()
end

instance = "$(@__DIR__)/../test/instances/pathological/chim_3_4_3.txt"
bench(instance)
