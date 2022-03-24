using LinearAlgebra
using MKL
using Base.Threads
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using CSV
using DataFrames

INSTANCE_DIR = "$(@__DIR__)/instances/chimera_droplets/512power"
OUTPUT_DIR = "$(@__DIR__)/results/512power"

disable_logging(LogLevel(1))
BLAS.set_num_threads(1)


function bench(loc_instance::String)
    id = threadid()

    m, n, t = 8, 8, 8

    L = n * m * t
    max_cl_states = 2 ^ (t-0)

    bond_dim = 32
    dE = 1

    num_states = 1000
    betas = collect(2:2:14)

    data = DataFrame(
        id = Any[],
        β = Any[],
        Layout = Any[],
        transform = Any[],
        energies = Any[],
        probability = Any[],
        largest_discarded_probability = Any[],
        statistic = Any[],
        time = Any[]
    )

    instance = joinpath(INSTANCE_DIR, loc_instance)
    out_path = joinpath(OUTPUT_DIR, "$id.csv")

    CSV.write(out_path, data, delim = ';', append = false)

    fg = factor_graph(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    Gauge = NoUpdate

    for β ∈ betas, Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), transform ∈ all_lattice_transformations

            δp = 1E-5 * exp(-β * dE)
            search_params = SearchParameters(num_states, δp)

            data = try
                net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], params)
                times = @elapsed sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

                data = DataFrame(
                    :id => loc_instance,
                    :β => β,
                    :Layout => Layout,
                    :transform => transform,
                    :energies => sol.energies[1:1],
                    :probability => sol.probabilities,
                    :largest_discarded_probability => sol.largest_discarded_probability,
                    :statistic => maximum(values(ctr.statistics)),
                    :time => times
                )
                data

            catch e
                data = DataFrame(
                    :id => loc_instance,
                    :β => β,
                    :Layout => Layout,
                    :transform => transform,
                    :energies => e,
                    :probability => "",
                    :largest_discarded_probability => "",
                    :statistic => "",
                    :time => ""
                )
                data
            end

            CSV.write(out_path, data, delim = ';', append = true)
            println(data)
        end
    end
end


@info "Benchmarking Started"

all_instances = readdir(INSTANCE_DIR, join=false)
@threads for idx ∈ 1:length(all_instances)
    instance = all_instances[idx]
    bench(instance)
end
