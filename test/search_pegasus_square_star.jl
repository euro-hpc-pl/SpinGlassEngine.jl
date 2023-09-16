using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

function bench(instance::String)
    m, n, t = 2, 2, 24

    max_cl_states = 2^8

    β = 2.0
    bond_dim = 64
    δp = 1E-4
    num_states = 20

    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        max_cl_states,
        spectrum = my_brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)
    Gauge = NoUpdate
    graduate_truncation = :graduate_truncate

    # Solve using PEPS search
    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), transform ∈ all_lattice_transformations
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), Sparsity ∈ (Dense, )
            net = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, cl_h, transform)
            ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], graduate_truncation, params; onGPU=onGPU)
            sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
            push!(energies, sol_peps.energies)
            clear_memoize_cache()
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

# best ground found: -59.65625
bench("$(@__DIR__)/instances/pegasus_droplets/2_2_3_00.txt")
