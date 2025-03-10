using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using MetaGraphs
using Graphs

function bench(instance::String)
    m, n, t = 3, 3, 3
    ig = prune(ising_graph(instance))
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = pegasus_lattice((m, n, t)),
    )

    println("ISING")
    num_vertices = nv(ig)
    println("Number of spins in the Ising graph: $num_vertices")

    total_spins = 0  
    println("POTTS")
    for v in vertices(potts_h)
        num_states_v = length(get_prop(potts_h, v, :spectrum).states)
        num_spins_v = num_states_v == 0 ? 0 : round(Int, log2(num_states_v))
        println("vertex ", v, " number of spins: ", num_spins_v)
        total_spins += num_spins_v
    end

    println("Total number of spins in the problem (Potts): ", total_spins)

end

bench("$(@__DIR__)/instances/embedded_tile_planting_simplified/tile_planting_2D_L_10_p1_0.0_p2_1.0_p3_0.0_inst_1_P4.txt")
bench("$(@__DIR__)/instances/embedded_tile_planting_simplified/tile_planting_2D_L_10_p1_0.0_p2_1.0_p3_0.0_inst_1_P4_nonzero.txt")
