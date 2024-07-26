using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine


function bench(instance::String)
    m, n, t = 16, 16, 8

    max_cl_states = 2^(t - 0)

    ground_energy = -3336.773383

    β = 3.0
    bond_dim = 32
    dE = 3.0
    δp = exp(-β * dE)
    num_states = 500
    all_betas = [β / 8, β / 4, β / 2, β]

    potts_h = potts_hamiltonian(
        ising_graph(instance),
        max_cl_states,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(; bd = bond_dim, ϵ = 1E-8, sw = 4, ts = 1E-16)
    search_params = SearchParameters(; max_states = num_states, cut_off_prob = δp)

    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, Zipper), Sparsity ∈ (Dense, Sparse)
        for Gauge ∈ (NoUpdate, GaugeStrategy, GaugeStrategyWithBalancing)
            for Layout ∈ (GaugesEnergy,), transform ∈ all_lattice_transformations

                net = PEPSNetwork{SquareSingleNode{Layout},Sparsity,Float64}(
                    m,
                    n,
                    potts_h,
                    transform,
                )
                ctr = MpsContractor{Strategy,Gauge,Float64}(
                    net,
                    params;
                    onGPU = onGPU,
                    βs = all_betas,
                    graduate_truncation = :graduate_truncate,
                )
                sol, s = low_energy_spectrum(
                    ctr,
                    search_params,
                    merge_branches(ctr; merge_type = :nofit),
                )

                @test sol.energies[begin] ≈ ground_energy

                push!(energies, sol.energies)
                clear_memoize_cache()
            end
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/chimera_droplets/2048power/001.txt")
