using SpinGlassEngine

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end


function run_test(instance, m, n, t, tran)
    β = 1
    bond_dim = 64
    δp = 1e-10
    num_states = 512
    βs = [β/16, β/8, β/4, β/2, β]

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
    Strategy = SVDTruncate # MPSAnnealing # 
    Sparsity = Sparse
    # tran = rotation(0)
    Layout = GaugesEnergy # EnergyGauges
    Gauge = NoUpdate

    net = PEPSNetwork{Square2{Layout}, Sparsity}(m, n, fg, tran)
    net2 = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg2, tran)

    ctr = MpsContractor{Strategy, Gauge}(net, βs, :graduate_truncate, params)
    ctr2 = MpsContractor{Strategy, Gauge}(net2, βs, :graduate_truncate, params)

    sol = low_energy_spectrum(ctr, search_params) #, merge_branches(ctr))
    sol2 = low_energy_spectrum(ctr2, search_params) # , merge_branches(ctr2))

    ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
    @test sol.energies ≈ energy.(Ref(ig), ig_states)
    fg_states = decode_state.(Ref(net), sol.states)
    @test sol.energies ≈ energy.(Ref(fg), fg_states)

    #@test sol.energies ≈ sol2.energies
    @test sol.energies[1: div(num_states, 8)] ≈ sol2.energies[1: div(num_states, 8)]
    #@test sol.states == sol2.states

    norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
    exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))

    @test norm_prob ≈ exct_prob
    push!(energies, sol.energies)

    for ii ∈ 1 : ctr.peps.nrows + 1
        for jj = 1 : length(βs)
            ψ1 = mps(ctr, ii, jj)
            ψ2 = mps(ctr2, ii, jj)
            o = ψ1 * ψ2 / sqrt((ψ1 * ψ1) * (ψ2 * ψ2))
            println(" mps; tran, i = " , tran,  ii, " jj = ", jj)
            println(bond_dimensions(ψ1))
            println(bond_dimensions(ψ2))
            println(o)
            @test o ≈ 1.
        end
    end

    for ii ∈ 0 : ctr.peps.nrows
        for jj = 1 : length(βs)
            ψ1_top = mps_top(ctr, ii, jj)
            ψ2_top = mps_top(ctr2, ii, jj)
            o_top = ψ1_top * ψ2_top / sqrt((ψ1_top * ψ1_top) * (ψ2_top * ψ2_top))
            println(" mps_top; tran, i = " , tran, " ",  ii, " jj = ", jj)
            println(bond_dimensions(ψ1_top))
            println(bond_dimensions(ψ2_top))
            println(o_top)
            @test o_top ≈ 1.
        end
    end

    clear_memoize_cache()
end

# instance = "$(@__DIR__)/instances/pegasus_nondiag/3x2x1.txt"
# m, n, t = 3, 2, 1
# for tran ∈ all_lattice_transformations
#     run_test(instance, m, n, t, tran)
# end

# instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"
# m, n, t = 4, 4, 1
# run_test(instance, m, n, t)

instance = "$(@__DIR__)/instances/pathological/pegasus_nd_3_4_1.txt"
m, n, t = 3, 4, 1
for tran ∈ all_lattice_transformations
    run_test(instance, m, n, t, tran)
end