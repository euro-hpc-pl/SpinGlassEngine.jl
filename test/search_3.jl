@testset "Simplest possible system of two spins" begin
    #
    # ----------------- Ising model ------------------
    #
    # E = -1.0 * s1 * s2 + 0.5 * s1 + 0.75 * s2
    #
    # states   -> [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    # energies -> [-2.25, 0.25, 0.75, 1.25]
    #
    # -------------------------------------------------
    #         Grid
    #     A1    |    A2
    #           |
    #       1 - | - 2
    #       3 - | - 4
    # -------------------------------------------------

    # Model's parameters
    J12 = -1.0
    J13 = -1.0
    J34 = -0.5
    J24 = -0.6
    J14 = -1.0
    h1 = 0.5
    h2 = 0.75
    h3 = 0.0
    h4 = 0.0

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 1) => h1,
             (2, 2) => h2,
             (1, 3) => J13,
             (3, 4) => J34,
             (2, 4) => J24,
             (1, 4) => J14,
             (3, 3) => h3,
             (4, 4) => h4,
    )

    # control parameters
    m, n = 2, 2
    L = 4
    β = 1.
    num_states = 8

    # read in pure Ising
    ig = ising_graph(D)

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict((1, 1) => 2, (1, 2) => 2, (2, 1) => 2, (2, 2) =>2), 
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2), 3 => (2, 1), 4 => (2, 2)), 
    )

    # set parameters to contract exactely
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    for transform ∈ all_lattice_transformations
        peps = PEPSNetwork(m, n, fg, transform, β=β)
        cluster_to_spin = Dict((1, 1) => 1, (1, 2) => 2, (2, 1) => 3, (2, 2) => 4)
        #cluster_to_spin = Dict((1, 1) => 1,(1, 2) => 2)

        @testset "has properly built PEPS tensors given transformation $(transform)" begin

            # horizontal alignment - 1 row, 2 columns
            if peps.nrows == 1 && peps.ncols == 2
                @test !transform.flips_dimensions

                l, k = cluster_to_spin[peps.vertex_map((1, 1))], cluster_to_spin[peps.vertex_map((1, 2))]

                v1 = [exp(-β * D[l, l] * σ) for σ ∈ [-1, 1]]
                v2 = [exp(-β * D[k, k] * σ) for σ ∈ [-1, 1]]

                @cast A[_, _, r, _, σ] |= v1[σ] * p1[σ, r]
                en = e * p2 .- minimum(e)
                @cast B[l, _, _, _, σ] |= v2[σ] * exp.(-β * en)[l, σ]

                @reduce ρ[σ, η] := sum(l) A[1, 1, l, 1, σ] * B[l, 1, 1, 1, η]
                if l == 2 ρ = ρ' end

                expected = [peps_tensor(peps, 1, 1), peps_tensor(peps, 1, 2)]
                @test expected ≈ [A, B]

            # vertical alignment - 1 column, 2 rows
            elseif peps.nrows == 2 && peps.ncols == 1
                @test transform.flips_dimensions
                l, k = cluster_to_spin[peps.vertex_map((1, 1))], cluster_to_spin[peps.vertex_map((2, 1))]
                # l, k = peps.map[1, 1], peps.map[2, 1]

                v1 = [exp(-β * D[l, l] * σ) for σ ∈ [-1, 1]]
                v2 = [exp(-β * D[k, k] * σ) for σ ∈ [-1, 1]]

                @cast A[_, _, _, d, σ] |= v1[σ] * p1[σ, d]
                en = e * p2 .- minimum(e)
                @cast B[_, u, _, _, σ] |= v2[σ] * exp.(-β * en)[u, σ]

                @reduce ρ[σ, η] := sum(u) A[1, 1, 1, u, σ] * B[1, u, 1, 1, η]
                if l == 2 ρ = ρ' end

                @test peps_tensor(peps, 1, 1) ≈ A
                @test peps_tensor(peps, 2, 1) ≈ B
            end

            @testset "which produces correct Gibbs state" begin
                @test ϱ ≈ ρ / sum(ρ)
            end
        end

        # solve the problem using B & B
        sol = low_energy_spectrum(peps, num_states)

        @testset "has correct spectrum given transformation $(transform)" begin
             for (σ, η) ∈ zip(exact_spectrum.states, sol.states)
                 for i ∈ 1:peps.nrows, j ∈ 1:peps.ncols
                    v = j + peps.ncols * (i - 1)
                     # 1 --> -1 and 2 --> 1
                     @test (η[v] == 1 ? -1 : 1) == σ[v]
                end
             end

             @test sol.energies ≈ exact_spectrum.energies
             @test sol.largest_discarded_probability === -Inf
        end
    end
end