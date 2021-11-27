using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

# This benchmark is meant to demonstrate minimal (or close) parameters
# (for a given Chimera instance) for which the ground state is found.
function bench()
    m = 16
    n = 16
    t = 8
    L = n * m * t

    ground_energy = -3336.773383

    β = 3.0
    bond_dim = 8
    δp = 1E-2
    num_states = 50

    instance = "$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    network = PEPSNetwork{Square{EnergyGauges}, Sparse}(m, n, fg, rotation(0))
    ctr = MpsContractor{MPSAnnealing}(network, [β], params)

    @time sol = low_energy_spectrum(ctr, search_params, merge_branches(network))

    @test sol.energies[begin] ≈ ground_energy
end

bench()
