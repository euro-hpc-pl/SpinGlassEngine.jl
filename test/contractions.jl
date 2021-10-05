#=
Problems are probably with mps and compresions, but hard to tell now.
=#



@testset "Contactions of Mpo and Mps for easy instances" begin
    m = 2
    n = 2
    t = 1

    β = 1.

    L = n * m * t

    #instance = "$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt"
    instance = "$(@__DIR__)/instances/basic/4_001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    for transform ∈ all_lattice_transformations
        peps = PEPSNetwork(m, n, fg, transform, β=β)
        for i in reverse(1:peps.nrows)
            
            println("row = ", i)
            W = mpo(peps, peps.mpo_main, i)
            
            for (j, dict) in W
                println(size.(values(dict)))
            end
            
            M = mps(peps, i) 

            #TODO dot(W,M) test

            #println(W.sites)
            #println(W.tensors)
        end
    end
end