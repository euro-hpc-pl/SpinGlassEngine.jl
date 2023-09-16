using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using SpinGlassExhaustive
using Logging
using Profile, PProf
using FlameGraphs

disable_logging(LogLevel(1))
Profile.init(n = 10^10, delay = 0.01)

function brute_force_gpu(ig::IsingGraph; num_states::Int)
    brute_force(ig, :GPU, num_states=num_states)
end


function bench(instance::String)
    m = 3
    n = 3
    t = 3

    β = 3
    bond_dim = 4
    δp = 1e-10
    num_states = 128
    println("creating factor graph" )
    @time begin
    ig = ising_graph(instance)

    cl_h = clustered_hamiltonian(
    ig,
    spectrum=brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
    )
    end
    params = MpsParameters(bond_dim, 1E-2, 1)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    Strategy = MPSAnnealing # SVDTruncate
    Sparsity = Sparse #Dense
    tran =  rotation(0)
    Layout = GaugesEnergy
    Gauge = NoUpdate
    println("creating network and contractor")
    @time begin
    net = PEPSNetwork{PegasusSquare, Sparsity}(m, n, cl_h, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)
    end
    println("solving")
    @time sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
end

instance = "$(@__DIR__)/../test/instances/pegasus_nondiag/test/003.txt"

bench(instance)
@profile bench(instance)

pprof(flamegraph(); webhost = "localhost", webport = 57328)
