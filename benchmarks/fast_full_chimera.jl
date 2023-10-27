using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

# This benchmark is meant to demonstrate minimal (or close) parameters
# (for a given Chimera instance) for which the true ground state is found.
function bench(instance::String, size::NTuple{3, Int})
    m, n, t = size
    β = 3.0
    δp = 1E-2
    bond_dim = 16
    max_cl_states = 2^8
    num_states = 1000
    max_sweeps = 10
    var_epsilon = 1E-3

    cl_h = clustered_hamiltonian(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice(size)
    )
    params = MpsParameters(bond_dim, var_epsilon, max_sweeps)
    search_params = SearchParameters(num_states, δp)

    net = PEPSNetwork{SquareSingleNode{EnergyGauges}, Dense}(m, n, cl_h, rotation(0))
    ctr = MpsContractor{MPSAnnealing}(net, [β], :graduate_truncate, params)
    @time sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    sol.energies[begin]
end

ground = -3336.773383
size = (16, 16, 8)
instance = "$(@__DIR__)/../test/instances/chimera_droplets/2048power/001.txt"

en1 = bench(instance, size)
en2 = bench(instance, size)

println(en1)
@assert en1 ≈ en2 #≈ ground
