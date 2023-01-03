using SpinGlassEngine
using Test

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end


function run_test(instance, m, n, t)
    β = 2
    bond_dim = 16
    δp = 1e-10
    num_states = 512

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum, #_gpu, # rm _gpu to use CPU
        cluster_assignment_rule=pegasus_lattice((m, n, t))
    )
    fg2 = factor_graph(
        ig,
        spectrum=full_spectrum, #_gpu, # rm _gpu to use CPU
        cluster_assignment_rule=super_square_lattice((m, n, 8))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)
    energies = []
    Gauge = NoUpdate
    βs = [β/16, β/8, β/4, β/2, β]

    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), Sparsity ∈ ( Sparse, )  # Dense,
        for Layout ∈ (EnergyGauges, GaugesEnergy)
            for tran ∈ all_lattice_transformations

                net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
                net2 = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg2, tran)

                ctr = MpsContractor{Strategy, Gauge}(net, βs, :graduate_truncate, params)
                ctr2 = MpsContractor{Strategy, Gauge}(net2, βs, :graduate_truncate, params)

                sol = low_energy_spectrum(ctr, search_params) #, merge_branches(ctr))
                sol2 = low_energy_spectrum(ctr2, search_params) #, merge_branches(ctr2))

                # ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
                # @test sol.energies ≈ energy.(Ref(ig), ig_states)
                # fg_states = decode_state.(Ref(net), sol.states)
                # @test sol.energies ≈ energy.(Ref(fg), fg_states)

                @test sol.energies[1: div(num_states, 2)] ≈ sol2.energies[1: div(num_states, 2)]
                #@test sol.states == sol2.states

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
                @test norm_prob ≈ exct_prob

                push!(energies, sol.energies[1])

                norm_prob = exp.(sol2.probabilities .- sol2.probabilities[1])
                exct_prob = exp.(-β .* (sol2.energies .- sol2.energies[1]))
                @test norm_prob ≈ exct_prob

                println("Eng = ", sol.energies[1])

                for ii ∈ 1 : ctr.peps.nrows + 1, jj ∈ 1 : length(βs)
                    ψ1, ψ2 = mps(ctr, ii, jj), mps(ctr2, ii, jj)
                    o = ψ1 * ψ2 / sqrt((ψ1 * ψ1) * (ψ2 * ψ2))
                    @test o ≈ 1.
                end
                for ii ∈ 0 : ctr.peps.nrows, jj ∈ 1 : length(βs)
                    ψ1_top, ψ2_top = mps_top(ctr, ii, jj),  mps_top(ctr2, ii, jj)
                    o_top = ψ1_top * ψ2_top / sqrt((ψ1_top * ψ1_top) * (ψ2_top * ψ2_top))
                    @test o_top ≈ 1.
                end
                clear_memoize_cache()
            end
        end
    end
    println("length energies ", length(energies))
    @test all(e -> e ≈ first(energies), energies)
end


instance = "$(@__DIR__)/instances/pathological/pegasus_3_4_1.txt"
m, n, t = 3, 4, 1
run_test(instance, m, n, t)
