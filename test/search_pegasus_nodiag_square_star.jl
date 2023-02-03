using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

function bench(instance::String)
    m, n, t = 4, 4, 24

    max_cl_states = 2^4

    β = 1.
    bond_dim = 16
    δp = 1E-4
    num_states = 1000

    ig = ising_graph(instance)
    fg, lp = factor_graph(
        ig,
        max_cl_states,
        spectrum=my_brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), transform ∈ all_lattice_transformations
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), Sparsity ∈ (Dense, )
            net = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg, lp, transform)
            ctr = MpsContractor{Strategy, NoUpdate}(net, [β/8, β/4, β/2, β], :graduate_truncate, params; onGPU=onGPU)
            sol_peps = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
            push!(energies, sol_peps.energies)
            clear_memoize_cache()
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

# best ground found for max_cl_states = 2^4: -537.0007291019799
bench("$(@__DIR__)/instances/pegasus_nondiag/4x4.txt")
