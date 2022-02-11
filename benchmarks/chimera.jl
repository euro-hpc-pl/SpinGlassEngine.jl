using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using CSV
using DataFrames


disable_logging(LogLevel(1))


function bench(instance_dir::String, out_path::String)
    m = 16
    n = 16
    t = 8
    L = n * m * t

    δp = 1E-2
    bond_dim = 16
    num_states = 100
    max_num_sweeps = 4
    variational_tol = 1E-8
    betas = collect(1:8)

    dir = cd(instance_dir)
    count = 0
    for (i, instance) ∈ enumerate(readdir(join=true))
        ig = ising_graph(instance)

        fg = factor_graph(
            ig,
            spectrum=brute_force,
            cluster_assignment_rule=super_square_lattice((m, n, t))
        )

        params = MpsParameters(bond_dim, variational_tol, max_num_sweeps)
        search_params = SearchParameters(num_states, δp)

        for β ∈ betas, Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, Sparse)
            for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), transform ∈ rotation.([0])
                  
                  
                data = try

                    net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                    ctr = MpsContractor{Strategy}(net, [β], params)
                    times = @elapsed sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

                    data = DataFrame(:i => i, :β => β, :Strategy => Strategy,
                    :Sparsity => Sparsity, :Layout => Layout,
                    :transform => transform, :energies => sol.energies[1:1],
                    :time => times)

                    data

                catch e 
                    

                    data = DataFrame(:i => i, :β => β, :Strategy => Strategy,
                    :Sparsity => Sparsity, :Layout => Layout,
                    :transform => transform, :energies => "ERROR",
                    :time => "ERROR")
                        
                    data
                    

                
                end
                println(data)
                CSV.write(out_path, data, delim = ';', append = count != 0)
                
                count += 1

                
            end
        end
    end
end


bench(
    "/home/tsmierzchalski/tensor/chimera2048_spinglass_power",
    "/home/tsmierzchalski/tensor/chimera2048_97.csv"
    )
