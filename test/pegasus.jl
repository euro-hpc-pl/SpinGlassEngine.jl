@testset "fuse_projectors correctly fuses projectors - simple case" begin
    projectors = ([1 1; 0 1], [1 0; 1 1])
    fused_expected, energy = rank_reveal(hcat(projectors...), :PE)
    @test fused_expected * energy == hcat(projectors[1], projectors[2])
    fused, transitions = fuse_projectors(projectors)
    @test hcat(transitions...) == energy
end

@testset "projectors_with_fusing correctly fuses tensors for a given network" begin
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

    for transform ∈ all_lattice_transformations
        peps = PEPSNetwork(m, n, fg, transform, β=β)
        cluster_to_spin = Dict((1, 1) => 1, (1, 2) => 2)
        #cluster_to_spin = Dict((1, 1) => 1,(1, 2) => 2)

        @testset "Projectors with fusing are build correctly for given transformation $(transform)" begin
            for v in vertices(fg)
                (pl, pb, pr, pt) = projectors_with_fusing(peps, v)
                println("pl", pl)
                println("pb", pb)
                println("pr", pr)
                println("pt", pt)
            end
        end
    end

end 