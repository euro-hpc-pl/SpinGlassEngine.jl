using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using CSV

function bench()
    m = 16
    n = 16
    t = 8
    L = n * m * t

    bond_dim = 128
    δp = 1E-2
    num_states = 100
    variational_tol = 1E-8
    max_num_sweeps = 4
    
    open("/home/tsmierzchalski/tensor/chimera.csv", "w") do file

        for f ∈ cd(readdir, "/home/tsmierzchalski/tensor/chimera2048_spinglass_power")
            if f == "groundstates_otn2d.txt" continue end
            
            instance = "/home/tsmierzchalski/tensor/chimera2048_spinglass_power/$(f)"
            ig = ising_graph(instance)

            fg = factor_graph(
                ig,
                #spectrum=full_spectrum,
                cluster_assignment_rule=super_square_lattice((m, n, t))
            )



            for β ∈ [1., 2., 3., 4., 5., 6., 7., 8.]

            
                params = MpsParameters(bond_dim, variational_tol, max_num_sweeps)
                search_params = SearchParameters(num_states, δp)

                for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, Sparse)
                    for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), transform ∈ rotation.([0])
                        #write(file, i, β, Strategy, Sparsity, Layout, transform)

                        network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                        ctr = MpsContractor{Strategy}(network, [β], params)

                        sol = low_energy_spectrum(ctr, search_params, merge_branches(network))

                        data = Dict("i" => i, "β" =>β, "Strategy" => Strategy, "Sparsity" => Sparsity, "Layout" => Layout, "transform" => transform, "energies" => sol.energies[1:1])
                        CSV.write(file, data)
                    end
                end
            end
        end
    end
end

bench()
