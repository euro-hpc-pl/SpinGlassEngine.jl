@testset "Canonise correctly truncates the state" begin
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t

    instance = "$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt"
    #instance = "$(@__DIR__)/instances/basic/4_001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    for transform ∈ all_lattice_transformations
        peps = PEPSNetwork(m, n, fg, transform, β=β)
        #println("transform ", transform)
        for i in reverse(1:peps.nrows)
            #println("row ", i)
            W1 = mpo(peps, peps.mpo_main, i)
            println("OVERLAP MPS") 
            M1 = mps(peps, i+1)
            ψ1 = dot(W1, M1)
            ψ2 = copy(ψ1)
            println("OVERLAP WITHOUT CANONISATION") 

            overlap1 = compress!(ψ1, W1, M1, peps.bond_dim, peps.var_tol, peps.sweeps)
            println("overlap after ", overlap1)
            
            #truncate!(ψ1, :left)
            canonise!(ψ2, :right)
            canonise!(ψ2, :left)

            println("OVERLAP WITH CANONISATION") 
            overlap2 = compress!(ψ2, W1, M1, peps.bond_dim, peps.var_tol, peps.sweeps)
            println("overlap after ", overlap2)

            #@test overlap1 < overlap2

            #println("--------------------------")

        end
    end
end