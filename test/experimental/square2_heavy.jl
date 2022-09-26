using SpinGlassExhaustive
using Logging
using Profile, PProf
using FlameGraphs

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end


#disable_logging(LogLevel(1))
Profile.init(n = 10^10, delay = 0.0001)

function bench(instance::String)

    m = 4
    n = 4
    t = 3

    β = 3
    bond_dim = 4
    δp = 1e-10
    num_states = 10

    ig = ising_graph(instance)
    #ig = ising_graph("$(@__DIR__)/../instances/chimera_droplets/128power/001.txt")
    #ig = ising_graph("$(@__DIR__)/../instances/pathological/pegasus_nd_3_4_1.txt")

    @time begin
    fg = factor_graph(
        ig,
        spectrum= brute_force_gpu, #rm _gpu to use CPU
        cluster_assignment_rule=pegasus_lattice((m, n, t))
    )
    end

    params = MpsParameters(bond_dim, 1E-8, 2)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    Strategy = MPSAnnealing # SVDTruncate
    Sparsity = Sparse #Dense
    tran =  rotation(0)
    Layout = EnergyGauges
    Gauge = NoUpdate

    net = PEPSNetwork{Square2{Layout}, Sparsity}(m, n, fg, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β/4, β/2, β], :graduate_truncate, params)

    sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    sol
end

#=
ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
@test sol.energies ≈ energy.(Ref(ig), ig_states)
fg_states = decode_state.(Ref(net), sol.states)
@test sol.energies ≈ energy.(Ref(fg), fg_states)

norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
@test norm_prob ≈ exct_prob

=#
instance = "$(@__DIR__)/../instances/pegasus_nondiag/pegasus_nd_4x4x3.txt"

sol = bench(instance)
println(sol.energies)

@profile bench(instance)

pprof(flamegraph(); webhost = "localhost", webport = 57328)

clear_memoize_cache()
