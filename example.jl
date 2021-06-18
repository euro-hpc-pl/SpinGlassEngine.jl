using Logging
using SpinGlassNetworks
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

function check_projectors_with_fusing(instance_path, nrows, ncols; β, sweeps, bond_dim, max_states)
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
    for v in vertices(fg)
        projectors_with_fusing(network, v)
    end
end

function check_fuse_projectors(projectors)
    fuse_projectors(projectors)
end
