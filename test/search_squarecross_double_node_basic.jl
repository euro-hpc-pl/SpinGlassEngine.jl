using SpinGlassEngine
using Test

function run_test_squarecross_double_node(instance, m, n, t)
    β = 2
    bond_dim = 16
    δp = 1e-10
    num_states = 512

    ig = ising_graph(instance)

    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum, #_gpu, # rm _gpu to use CPU
        cluster_assignment_rule = pegasus_lattice((m, n, t)),
    )
    potts_h2 = potts_hamiltonian(
        ig,
        spectrum = full_spectrum, #_gpu, # rm _gpu to use CPU
        cluster_assignment_rule = super_square_lattice((m, n, 8)),
    )

    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cut_off_prob = δp)
    energies = []
    Gauge = NoUpdate
    βs = [β / 16, β / 8, β / 4, β / 2, β]

    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), Sparsity ∈ (Sparse,)
        for Layout ∈ (EnergyGauges, GaugesEnergy)
            for tran ∈ all_lattice_transformations

                net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity,Float64}(
                    m,
                    n,
                    potts_h,
                    tran,
                )
                net2 = PEPSNetwork{KingSingleNode{Layout},Sparsity,Float64}(
                    m,
                    n,
                    potts_h2,
                    tran,
                )

                ctr = MpsContractor{Strategy,Gauge,Float64}(
                    net,
                    params;
                    onGPU = onGPU,
                    βs = βs,
                    graduate_truncation = :graduate_truncate,
                )
                ctr2 = MpsContractor{Strategy,Gauge,Float64}(
                    net2,
                    params;
                    onGPU = onGPU,
                    βs = βs,
                    graduate_truncation = :graduate_truncate,
                )

                sol, s = low_energy_spectrum(ctr, search_params) #, merge_branches(ctr))
                sol2, s = low_energy_spectrum(ctr2, search_params) #, merge_branches(ctr2))

                # ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                # @test sol.energies ≈ energy.(Ref(ig), ig_states)
                # potts_h_states = decode_state.(Ref(net), sol.states)
                # @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

                @test sol.energies[1:div(num_states, 2)] ≈
                      sol2.energies[1:div(num_states, 2)]
                #@test sol.states == sol2.states

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
                @test norm_prob ≈ exct_prob

                push!(energies, sol.energies[1])

                norm_prob = exp.(sol2.probabilities .- sol2.probabilities[1])
                exct_prob = exp.(-β .* (sol2.energies .- sol2.energies[1]))
                @test norm_prob ≈ exct_prob

                for ii ∈ 1:ctr.peps.nrows+1, jj ∈ 1:length(βs)
                    ψ1, ψ2 = mps(ctr, ii, jj), mps(ctr2, ii, jj)
                    o = ψ1 * ψ2 / sqrt((ψ1 * ψ1) * (ψ2 * ψ2))
                    @test o ≈ 1.0
                end
                for ii ∈ 0:ctr.peps.nrows, jj ∈ 1:length(βs)
                    ψ1_top, ψ2_top = mps_top(ctr, ii, jj), mps_top(ctr2, ii, jj)
                    o_top = ψ1_top * ψ2_top / sqrt((ψ1_top * ψ1_top) * (ψ2_top * ψ2_top))
                    @test o_top ≈ 1.0
                end
                clear_memoize_cache()
            end
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end


instance = "$(@__DIR__)/instances/pathological/pegasus_3_4_1.txt"
m, n, t = 3, 4, 1
run_test_squarecross_double_node(instance, m, n, t)
