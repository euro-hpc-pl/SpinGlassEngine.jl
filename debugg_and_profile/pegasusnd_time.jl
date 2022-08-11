using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using SpinGlassExhaustive
using Logging
using Profile, PProf
using FlameGraphs

disable_logging(LogLevel(1))
Profile.init(n = 10^8, delay = 0.01)

function brute_force_gpu(ig::IsingGraph; num_states::Int)
    brute_force(ig, :GPU, num_states=num_states)
end


function bench(instance::String)
    m = 4
    n = 4
    t = 3

    β = 3
    bond_dim = 4
    δp = 1e-10
    num_states = 10

    ig = ising_graph(instance)

    fg = factor_graph(
    ig,
    spectrum=brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-2, 1)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    Strategy = MPSAnnealing # SVDTruncate
    Sparsity = Sparse #Dense
    tran =  rotation(0)
    Layout = GaugesEnergy
    Gauge = NoUpdate

    net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)

    @time sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
end

instance = "$(@__DIR__)/../test/instances/pegasus_droplets/4_4_3_00.txt"

bench(instance)
@profile bench(instance)

pprof(flamegraph(); webhost = "localhost", webport = 57328)
