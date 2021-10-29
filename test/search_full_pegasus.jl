using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench()
    m = 3
    n = 3
    t = 24

    β = 3.
    schedule = 1.
    states_to_keep = 2

    instance = "$(@__DIR__)/instances/pegasus_droplets/$(m)_$(n)_3_00.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    #for transform ∈ all_lattice_transformations
    for transform ∈ rotation.([0])
        peps = FusedNetwork(m, n, fg, transform, β=β)
        update_gauges!(peps, :rand)
        map = peps.tensors_map

        f = open("pegasus3_3.txt", "w") 
        write(f, " transform ") + write(f, string(transform))


        for j in [:virtual, :site, :central_d, :central_v, :gauge_h]
            write(f, " type of tensor ") + write(f, string(j))
            key_value = [k for (k,v) in map if v==j]
            for i in collect(sort(key_value))
                write(f, " site ") + write(f, string(i)) + write(f, " size ") + write(f, string(tensor_size(peps, i)))
            end
            write(f, "------------")
        end
        close(f)

        #=
        map = peps.tensors_map

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
        #sol = low_energy_spectrum(peps, states_to_keep)#, merge_branches(peps))
        #println(sol.energies[1:1])
    end
end

bench()
