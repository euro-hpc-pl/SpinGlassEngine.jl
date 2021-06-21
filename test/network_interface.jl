@testset "fuse_projectors correctly fuses projectors." begin
    projectors = ([1 1; 0 1], [1 0; 1 1])
    expected_fused = [1 0; 0 1]
    expected_transitions = [[1 1; 0 1], [1 0; 1 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end


@testset "projectors_with_fusing correctly fuses tensors for a given network" begin
    # Model's parameters
    J12 = -1.0
    h1 = 0.5
    h2 = 0.75

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 1) => h1,
             (2, 2) => h2,
    )

    # control parameters
    m, n = 1, 2
    L = 2
    β = 1.
    num_states = 4

    # read in pure Ising
    ig = ising_graph(D)

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict((1, 1) => 2, (1, 2) => 2),
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2)), # treat it as a grid with 1 spin cells
    )

    # set parameters to contract exactely
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    # get BF results for comparison
    exact_spectrum = brute_force(ig; num_states=num_states)
    ϱ = gibbs_tensor(ig, β)

    # split on the bond
    p1, e, p2 = get_prop.(Ref(fg), Ref((1, 1)), Ref((1, 2)), (:pl, :en, :pr))

    @testset "has correct energy on the bond" begin
        en = [ J12 * σ * η for σ ∈ [-1, 1], η ∈ [-1, 1]]
        @test en ≈ p1 * (e * p2)
        @test p1 ≈ p2 ≈ I
    end
    correct_transform = (LatticeTransformation((1, 2, 3, 4), true), LatticeTransformation((3, 4, 1, 2), true), 
                        LatticeTransformation((2, 1, 4, 3), true), LatticeTransformation((4, 3, 2, 1), true))
    for transform ∈ correct_transform
        # fails for transformations ((4, 1, 2, 3), (2, 3, 4, 1), (1, 4, 3, 2), (3, 2, 1, 4))
        peps = PEPSNetwork(m, n, fg, transform, β=β)
        @testset "Projectors with fusing are built correctly for given transformation $(transform)" begin
            pl, pb, pr, pt = [], [], [], []
            for v in vertices(fg)
                (pl_new, pb_new, pr_new, pt_new) = projectors_with_fusing(peps, v)
                push!(pl, pl_new)
                push!(pb, pb_new)
                push!(pr, pr_new)
                push!(pt, pt_new)
            end
            @test pl[1] == pr[2]
            @test pt[1] == pt[2]
            @test pb[1] == pb[2]
        end
        @testset "Tensors with fusing are built correctly for given transformation $(transform)" begin
            for v in vertices(fg)
                A = build_tensor_with_fusing(peps, v)
            end
        end
        
    end

end
