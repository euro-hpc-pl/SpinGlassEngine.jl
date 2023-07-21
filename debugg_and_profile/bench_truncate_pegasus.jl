using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using SpinGlassExhaustive
using Logging
using Profile, PProf
using FlameGraphs
using CUDA
using Memoization

disable_logging(LogLevel(1))
Profile.init(n = 10^9, delay = 0.01)

function brute_force_gpu(ig::IsingGraph; num_states::Int)
    brute_force(ig, :GPU, num_states=num_states)
end

onGPU = true

function bench(instance::String)
    m = 7
    n = 7
    t = 3
    β = 0.5
    bond_dim = 8
    δp = 1E-6
    num_states = 64
    iter = 1
    println("creating factor graph" )
    cs = 1024

    @time begin
    ig = ising_graph(instance)
    fg = factor_graph(
        ig,
        spectrum=full_spectrum,  # brute_force_gpu, # rm _gpu to use CPU
        cluster_assignment_rule = pegasus_lattice((m, n, t))
    )
    end
    @time fg2 = truncate_factor_graph_2site_BP(fg, cs; beta = β, iter=iter)
end

instance = "$(@__DIR__)/../test/instances/pegasus_random/P8/CBFM-P/SpinGlass/001_sg.txt"
# bench(instance)
@profile bench(instance)
pprof(flamegraph(); webhost = "localhost", webport = 57320)
