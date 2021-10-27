@testset "Pegasus-like instance has the correct ground state energy" begin

    ground_energy = -23.301855

    m = 3
    n = 4
    t = 3
    
    β = 1.

    schedule = 1.

    L = n * m * t
    states_to_keep = 20

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t)) 
    )

    for transform ∈ all_lattice_transformations
        peps = FusedNetwork(m, n, fg, transform, β=β)
        update_gauges!(peps, :rand)
        sol = low_energy_spectrum(peps, states_to_keep, merge_branches(peps))
        @test first(sol.energies) ≈ ground_energy
        map = peps.tensors_map

        f = open("cross.txt", "w") # do this once
        write(f, " transform ") + write(f, string(transform))


        for j in [:virtual, :central_d, :central_v, :gauge_h, :site]
            write(f, " type of tensor ") + write(f, string(j))
            key_value = [k for (k,v) in map if v==j]
            for i in collect(sort(key_value))
                write(f, " site ") + write(f, string(i)) + write(f, " size ") + write(f, string(size(tensor(peps, i))))
            end
            write(f, "------------")
        end
        close(f)
#=
        for j in [:virtual, :central_d, :central_v, :gauge_h, :site]
            println("type of tensor ", j)
            key_value = [k for (k,v) in map if v==j]
            for i in collect(sort(key_value))
                println("site ", i, " size ", size(tensor(peps, i)))
            end
            println("-------------------")
            println("-------------------")
            println("-------------------")
        end
        =#
    
            
    end
end