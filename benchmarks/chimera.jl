using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using CSV

disable_logging(LogLevel(1))

function bench(instance_dir::String, outer_dir::String)
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

    open(outer_dir, "w") do file

    for (i, instance) ∈ enumerate(readdir(join=true))
        println(instance)
        ig = ising_graph(instance)

        fg = factor_graph(
            ig,
            #spectrum=full_spectrum,
            cluster_assignment_rule=super_square_lattice((m, n, t))
        )

        params = MpsParameters(bond_dim, variational_tol, max_num_sweeps)
        search_params = SearchParameters(num_states, δp)

        for β ∈ betas, Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
            for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), trans ∈ rotation.([0])

                network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, trans)
                ctr = MpsContractor{Strategy}(network, [β], params)

                sol = low_energy_spectrum(ctr, search_params, merge_branches(network))
                data = Dict(
                    "i" => i, "β" => β, "Strategy" => Strategy,
                    "Sparsity" => Sparsity, "Layout" => Layout,
                    "transform" => trans, "energies" => sol.energies[1:1]
                )
                CSV.write(file, data)
            end
        end
    end
end
end

bench(
    "/home/bartek/Desktop/Chimera/chimera2048_spinglass_power",
    "/home/bartek/Desktop/Chimera/chimera.csv"
)
