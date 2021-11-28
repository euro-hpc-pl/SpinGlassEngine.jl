using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

# This benchmark is meant to demonstrate minimal (or close) parameters
# (for a given Chimera instance) for which the true ground state is found.
function bench()
    m = 16
    n = 16
    t = 8
    L = n * m * t

    ground_energy = -3336.773383

    β = 3.0
    δp = 1E-2
    bond_dim = 14
    num_states = 50
    max_sweeps = 4
    var_epsilon = 1E-3

    fg = factor_graph(
        ising_graph("$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"),
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, var_epsilon, max_sweeps)
    search_params = SearchParameters(num_states, δp)

    network = PEPSNetwork{Square{EnergyGauges}, Dense}(m, n, fg, rotation(0))
    ctr = MpsContractor{MPSAnnealing}(network, [β], params)
    @time sol = low_energy_spectrum(ctr, search_params, merge_branches(network))
    @test sol.energies[begin] ≈ ground_energy
end

bench()
