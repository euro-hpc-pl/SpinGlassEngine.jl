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


Dcut = 8
β = 0.5
tolV = 1E-16
tolS = 1E-16
max_sweeps = 0
indβ = 1
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
MAX_STATES = 128
METHOD = :psvd_sparse #:psvd_sparse #:svd
DE = 16.0
δp = 1E-5*exp(-β * DE)

# cl_h = clustered_hamiltonian(
#     ig,
#     spectrum=my_brute_force, #rm _gpu to use CPU
#     cluster_assignment_rule=pegasus_lattice((m, n, t))
# )


params = MpsParameters(Dcut, tolV, max_sweeps, tolS, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
search_params = SearchParameters(MAX_STATES, δp)

onGPU = true
Strategy = Zipper  # MPSAnnealing SVDTruncate
Layout = GaugesEnergy
Gauge = NoUpdate
cl_states = [2^10,]
tran = LatticeTransformation((1, 2, 3, 4), false)

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
    results_folder = "$(@__DIR__)/../test/instances/pegasus_random/P8/CBFM-P/SpinGlass/BP"
    inst = "001"
    @time begin
    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        spectrum=full_spectrum,  # brute_force_gpu, # rm _gpu to use CPU
        cluster_assignment_rule = pegasus_lattice((m, n, t))
    )
    end
    @time cl_h = truncate_clustered_hamiltonian(cl_h, β, cs, results_folder, inst; tol=1e-6, iter=iter)

    net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparse}(m, n, cl_h, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)
    @time sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    println("sol ", sol)
end

instance = "$(@__DIR__)/../test/instances/pegasus_random/P8/CBFM-P/SpinGlass/001_sg.txt"
# bench(instance)
@profile bench(instance)
pprof(flamegraph(); webhost = "localhost", webport = 57320)
