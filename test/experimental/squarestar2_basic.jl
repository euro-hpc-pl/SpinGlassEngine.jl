using SpinGlassEngine

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end


function run_test(instance, m, n, t, tran)
    β = 2
    bond_dim = 64
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

    # Solve using PEPS search
    energies = Vector{Float64}[]
    Strategy = MPSAnnealing #SVDTruncate
    Sparsity = Sparse
    # tran = rotation(0)
    Layout = GaugesEnergy
    Gauge = NoUpdate

    net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
    net2 = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg2, tran)

    ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)
    ctr2 = MpsContractor{Strategy, Gauge}(net2, [β/8, β/4, β/2, β], :graduate_truncate, params)

    # sol = low_energy_spectrum(ctr, search_params) #, merge_branches(ctr))
    # sol2 = low_energy_spectrum(ctr2, search_params) #, merge_branches(ctr2))

    # ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
    # @test sol.energies ≈ energy.(Ref(ig), ig_states)
    # fg_states = decode_state.(Ref(net), sol.states)
    # @test sol.energies ≈ energy.(Ref(fg), fg_states)

    # #@test sol.energies ≈ sol2.energies
    # @test sol.energies[1: div(num_states, 2)] ≈ sol2.energies[1: div(num_states, 2)]
    # #@test sol.states == sol2.states

    # norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
    # exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
    # @test norm_prob ≈ exct_prob

    # # push!(energies, sol.energies)

    # norm_prob = exp.(sol2.probabilities .- sol2.probabilities[1])
    # exct_prob = exp.(-β .* (sol2.energies .- sol2.energies[1]))
    # @test norm_prob ≈ exct_prob

    # println("Eng = ", sol.energies[1])

    for ii in 1 : m
        ψ1 = mps(ctr, ii + 1, 4)
        ψ1_top = mps_top(ctr, ii, 4)
        ψ2 = mps(ctr2, ii + 1, 4)
        ψ2_top = mps_top(ctr2, ii, 4)
        o = ψ1 * ψ2 / sqrt((ψ1 * ψ1) * (ψ2 * ψ2))
        o_top = ψ1_top * ψ2_top / sqrt((ψ1_top * ψ1_top) * (ψ2_top * ψ2_top))
        @test o ≈ 1.
        @test o_top ≈ 1.
    end

    clear_memoize_cache()
end


instance = "$(@__DIR__)/../instances/pathological/pegasus_3_4_1.txt"
m, n, t = 3, 4, 1
for tran ∈ all_lattice_transformations
    run_test(instance, m, n, t, tran)
end