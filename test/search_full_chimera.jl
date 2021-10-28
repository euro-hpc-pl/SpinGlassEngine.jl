using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench()
    m = 16 
    n = 16
    t = 8

    β = 3.

    L = n * m * t
    num_states = 100

    instance = "$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    #for transform ∈ all_lattice_transformations
    for transform ∈ rotation.([0])
        peps = PEPSNetwork(m, n, fg, transform, β=β, bond_dim=32)
        update_gauges!(peps, :rand)
        @time sol = low_energy_spectrum(peps, num_states, merge_branches(peps))
        println(sol.energies[1:1])


        map = peps.tensors_map

        f = open("chimera.txt", "w") # do this once
        write(f, " transform ") + write(f, string(transform))


        for j in [:virtual, :central_d, :central_v, :gauge_h, :site]
            write(f, " type of tensor ") + write(f, string(j))
            key_value = [k for (k,v) in map if v==j]
            for i in collect(sort(key_value))
                write(f, " site ") + write(f, string(i)) + write(f, " size ") + write(f, string(tensor_size(peps, i)))
            end
            write(f, "------------")
        end
        close(f)
        #=
        for j in [:virtual, :central_d, :central_v, :gauge_h, :site]
            println("type of tensor ", j)
            key_value = [k for (k,v) in map if v==j]
            for i in collect(sort(key_value))
                println("site ", i, " size ", tensor_size(peps, i))
            end
            println("-------------------")
            println("-------------------")
            println("-------------------")
        end=#
    end
end

bench()
