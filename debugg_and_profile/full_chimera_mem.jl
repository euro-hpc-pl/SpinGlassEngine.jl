using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Logging
using Profile

disable_logging(LogLevel(1))

function bench(instance::String)
    m = 8
    n = 8
    t = 8

    L = n * m * t
    max_cl_states = 2^(t-0)

    ground_energy = -3336.773383

    β = 2
    bond_dim = 32
    DE = 1.0
    δp = 1E-5 * exp(-β * DE)
    num_states = 1000

    cl_h = clustered_hamiltonian(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    net = PEPSNetwork{SquareSingleNode{EnergyGauges}, Dense}(m, n, cl_h, rotation(0))
    ctr = MpsContractor{SVDTruncate, NoUpdate}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)
    sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

    #@assert sol.energies[begin] ≈ ground_energy
end

#instance = "$(@__DIR__)/../test/instances/chimera_droplets/2048power/001.txt"
instance = "$(@__DIR__)/../test/instances/chimera_droplets/512power/001.txt"

bench(instance)
Profile.clear_malloc_data()
bench(instance)
exit()
