using MKL
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using CSV
using DataFrames


disable_logging(LogLevel(1))


function bench(instance_dir::String, out_path::String)
    m = 12
    n = 12
    t = 8
    
    L = n * m * t
    max_cl_states = 2^(t-0)

    bond_dim = 32
    dE = 5
    
    num_states = 10000
    betas = collect(2:2:14)

    count = 0

    for (i, instance) ∈ enumerate(readdir(instance_dir, join=true))

        fg = factor_graph(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t)))

        params = MpsParameters(bond_dim, 1E-8, 10)
        Gauge = NoUpdate
        
        for β ∈ betas, Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
            for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), transform ∈ all_lattice_transformations

                δp = 1E-5*exp(-β * dE)  
                search_params = SearchParameters(num_states, δp)

                data = try

                    net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                    ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], params)
                    times = @elapsed sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

                    data = DataFrame(:i => i, :β => β, :Layout => Layout,
                    :transform => transform, :energies => sol.energies[1:1],
                    :time => times)

                    data

                catch e 
                    

                    data = DataFrame(:i => i, :β => β, :Layout => Layout,
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

@info "Benchmarking Started"

bench(
    "$(@__DIR__)/instances/chimera_droplets/1152power",
    "$(@__DIR__)/chimera1152.csv"
    )

