@testset "Tensors have correct sizes" begin

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
        for i ∈ 1:peps.nrows-1, k ∈ 1:1//2:peps.ncols
            j = denominator(k) == 1 ? numerator(k) : k
            l, u, r, d = tensor_size(peps, (i+1//2, j))
            @test size(tensor(peps, (i+1//2, j))) == tensor_size(peps, (i+1//2, j))
        end
    end
end