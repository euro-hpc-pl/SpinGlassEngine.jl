using MKL
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using CSV
using DataFrames


disable_logging(LogLevel(1))


function bench(instance_dir::String, out_path::String)
    m = 8
    n = 8
    t = 8
    
    L = n * m * t
    max_cl_states = 2^(t-0)

    bond_dim = 32
    dE = 1
    
    num_states = 1000
    betas = collect(2:2:14)

    data = DataFrame(i = Any[], β = Any[], Layout = Any[] , transform = Any[], energies = Any[], probability = Any[], 
    largest_discarded_probability = Any[], statistic = Any[], time = Any[])

    CSV.write(out_path, data, delim = ';', append = false)


    for (i, instance) ∈ enumerate(readdir(instance_dir, join=false))

        cl_h = clustered_hamiltonian(
        ising_graph(instance_dir * "/" * instance),
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

                    net = PEPSNetwork{SquareSingleNode{Layout}, Sparsity}(m, n, cl_h, transform)
                    ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params)
                    times = @elapsed sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))


                    data = DataFrame(:i => instance, :β => β, :Layout => Layout,
                    :transform => transform, :energies => sol.energies[1:1], 
                    :probability => sol.probabilities, :largest_discarded_probability => sol.largest_discarded_probability,
                    :statistic => maximum(values(ctr.statistics)), :time => times)

                    data

                catch e 
                    
                    data = DataFrame(:i => instance, :β => β, :Layout => Layout,
                    :transform => transform, :energies => e, :probability => "", :largest_discarded_probability => "",
                    :statistic => "", :time => "")
                        
                    data
                
                end
                println(data)
                CSV.write(out_path, data, delim = ';', append = true)
                
                
            end
        end
    end

end

@info "Benchmarking Started"

bench(
    "$(@__DIR__)/instances/chimera_droplets/512power",
    "$(@__DIR__)/chimera512.csv"
    )

