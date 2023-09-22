using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using SpinGlassExhaustive
using Logging
using Profile, PProf
using FlameGraphs

disable_logging(LogLevel(1))
#Profile.init(n = 10^10, delay = 0.01)

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end


function bench()
    m = 8
    n = 8
    t = 4

    β = 1.
    Dcut = 4
    tolV = 0.
    tolS = 0.
    max_sweeps = 1
    indβ = 1

    ig = ising_graph("$(@__DIR__)/../test/instances/zephyr_random/Z4/RAU/SpinGlass/001_sg.txt")

    @time begin
        println("factor graph")
        cl_h = clustered_hamiltonian(
            ig,
            # max_cl_states,
            spectrum = full_spectrum,  #brute_force_gpu, # rm _gpu to use CPU
            cluster_assignment_rule = zephyr_lattice((m, n, t))
        )
    end
    println("Factor graph memory = ", format_bytes.(Base.summarysize(cl_h)))

    params = MpsParameters(Dcut, tolV, max_sweeps)
    Strategy = MPSAnnealing # SVDTruncate
    Sparsity = Sparse # Dense
    # tran =  LatticeTransformation((3, 4, 1, 2), false)
    tran =  LatticeTransformation((1, 2, 3, 4), false)
    Layout = GaugesEnergy
    Gauge = NoUpdate

    net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparsity}(m, n, cl_h, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)

    i = div(m, 2)
    @time begin
        println(" MPO ")
        W = SpinGlassEngine.mpo(ctr, ctr.layers.main, i, indβ)
    end
    println(device(W))
    println("Mpo memory = ", format_bytes.(measure_memory(W)))

    println("rand QMps")
    ψ = rand(QMps{Float64}, local_dims(W, :down), Dcut)
    println(device(ψ))
    println("canonnise")
    @time begin
        canonise!(ψ, :right)
        canonise!(ψ, :left)
    end

    ψ0 = rand(QMps{Float64}, local_dims(W, :up), Dcut)
    canonise!(ψ0, :right)
    canonise!(ψ0, :left)
    println(device(ψ0))

    println("variational")
    @time begin
        overlap, env = variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    end
    overlap

    @time begin
        println("zipper")
        ψ1 = zipper(W, ψ, method=:psvd_sparse, Dcut=Dcut, tol=tolS)
    end
    overlap
end

# println("pre-sweep")
# clear_memoize_cache()
# bench()

println("measuring ... ")
clear_memoize_cache()
@profile bench()

pprof(flamegraph(); webhost = "localhost", webport = 57328)
