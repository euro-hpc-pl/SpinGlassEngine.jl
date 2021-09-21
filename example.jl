using Logging
using SpinGlassEngine
using MetaGraphs
using LightGraphs
disable_logging(LogLevel(1))

function solve_instance(instance_path, nrows, ncols; β, sweeps, bond_dim, max_states)
    ig = ising_graph(instance_path)
    fg = factor_graph(
        ig,
        cluster_assignment_rule=super_square_lattice((nrows, ncols, 8))
    )
    network = PEPSNetwork(
        nrows,
        ncols,
        fg,
        rotation(0),
        β=β,
        bond_dim=bond_dim,
        sweeps=sweeps,
    )
    low_energy_spectrum(network, max_states, merge_branches(network, 1.0))
end



function solve_instance(instance_path, nrows, ncols, t; β, sweeps, bond_dim, max_states)
    ig = ising_graph(instance_path)
    fg = factor_graph(
        ig,
        cluster_assignment_rule=super_square_lattice((nrows, ncols, t))
    )
    network = FusedNetwork(
        nrows,
        ncols,
        fg,
        rotation(0),
        β=β,
        bond_dim=bond_dim,
        sweeps=sweeps,
    )
    low_energy_spectrum(network, max_states, merge_branches(network, 1.0))
end
