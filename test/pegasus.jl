@testset "fuse_projectors correctly fuses projectors - simple case" begin
    projectors = ([1 1; 0 1], [1 0; 1 1])
    fused_expected, energy = rank_reveal(hcat(projectors...), :PE)
    @test fused_expected * energy == hcat(projectors[1], projectors[2])
    fused, transitions = fuse_projectors(projectors)
    @test hcat(transitions...) == energy
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
                #println("pl", pl)
                #println("pb", pb)
                #println("pr", pr)
                #println("pt", pt)
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

#@testset "projectors_with_fusing correctly fuses tensors for a given network" begin 
    #
    # E = -1.0 * s1 * s2 + (-1.0) * s1 * s3 + 0.5 * s1 + 0.75 * s2 + 0.2 * s3
    #
    # states   -> [[-1, -1, -1], [1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, 1, -1], [1, -1, 1]]
    # energies -> [-3.45, -0.55, 1.55, -1.05, 0.05, 2.45, 1.05, -0.05]
    #
    # -------------------------------------------------
    #         Grid
    #     A1    |    A2
    #           |
    #       1 - | - 2
    #         \ |
    #          \|     
    #   --------|---------
    #           |\   A3 
    #           | \  
    #           |   3 
    # -------------------------------------------------

#    J12 = -1.0
#    J13 = -1.0
#    h1 = 0.5
#    h2 = 0.75
#    h3 = 0.2

    # dict to be read
#    D = Dict((1, 2) => J12,
#             (1, 3) => J13,
#             (1, 1) => h1,
#             (2, 2) => h2,
#             (3, 3) => h3,
#    )

    # control parameters
#    m, n = 2, 2
#    L = 2
#    β = 1.
#    num_states = 4

    # read in pure Ising
#    ig = ising_graph(D)

    # construct factor graph with no approx
#    fg = factor_graph(
#        ig,
#        Dict((1, 1) => 2, (1, 2) => 2, (1, 3) => 3),
#        spectrum = full_spectrum,
#        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2), 3 => (1, 3)), 
#    )

    # set parameters to contract exactely
#    control_params = Dict(
#        "bond_dim" => typemax(Int),
#        "var_tol" => 1E-8,
#        "sweeps" => 4.
#    )

    # get BF results for comparison
#    exact_spectrum = brute_force(ig; num_states=num_states)
#    ϱ = gibbs_tensor(ig, β)

    # split on the bond
#    p1, e12, p2 = get_prop.(Ref(fg), Ref((1, 1)), Ref((1, 2)), (:pl, :en, :pr))
#    p1d, e13, p3 = get_prop.(Ref(fg), Ref((1, 1)), Ref((1, 3)), (:pl, :en, :pr))

#    @testset "has correct energy on the bond" begin
#        en12 = [ J12 * σ * η for σ ∈ [-1, 1], η ∈ [-1, 1]]
#        en13 = [ J13 * σ * η for σ ∈ [-1, 1], η ∈ [-1, 1]]
#        @test en12 ≈ p1 * (e12 * p2)
#        @test p1 ≈ p2 ≈ I
#        @test en13 ≈ p1d * (e13 * p3)
#    end

#    for transform ∈ all_lattice_transformations
#        peps = PEPSNetwork(m, n, fg, transform, β=β)
        #cluster_to_spin = Dict((1, 1) => 1, (1, 2) => 2)

#        @testset "Projectors with fusing are built correctly for given transformation $(transform)" begin
#            for v in vertices(fg)
#                (pl, pb, pr, pt) = projectors_with_fusing(peps, v)
#                println("pl", pl)
#                println("pb", pb)
#                println("pr", pr)
#                println("pt", pt)
#            end
#        end
#    end

#end 