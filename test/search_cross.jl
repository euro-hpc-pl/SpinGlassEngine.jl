@testset "Pegasus-like instance has the correct ground state energy" begin

    ground_energy = -23.301855

    m = 3
    n = 4
    t = 3
    
    β = 1.

    L = n * m * t
    num_states = 22

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t)) 
    )

    params = MpsParameters()

    for Strategy ∈ (Basic,)
        for Sparsity ∈ (Dense,) #Sparse
            for Layout ∈ (EnergyGauges, GaugesEnergy)
                for transform ∈ all_lattice_transformations

                    network = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg, transform)
                    contractor = MpsContractor{Strategy}(network, [β], params)
                    sol = low_energy_spectrum(contractor, num_states)

                    @test first(sol.energies) ≈ ground_energy
                end
            end
        end
    end
end