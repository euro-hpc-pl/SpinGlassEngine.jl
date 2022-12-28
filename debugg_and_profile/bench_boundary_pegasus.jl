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

function bench()

    m = 7
    n = 7
    t = 3

    β = 1.
    Dcut = 4
    tolV = 0.
    tolS = 0.
    max_sweeps = 1
    indβ = 1

    ig = ising_graph("$(@__DIR__)/../test/instances/pegasus_random/P8/RAU/SpinGlass/001_sg.txt")

    fg = factor_graph(
        ig,
        spectrum= brute_force_gpu, #rm _gpu to use CPU
        cluster_assignment_rule=pegasus_lattice((m, n, t))
    )

    params = MpsParameters(Dcut, tolV, max_sweeps)
    Strategy = MPSAnnealing # SVDTruncate
    Sparsity = Sparse # Dense
    tran =  LatticeTransformation((3, 4, 1, 2), false)
    # tran =  LatticeTransformation((1, 2, 3, 4), false)
    Layout = GaugesEnergy
    Gauge = NoUpdate

    net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)

    i = 4
    println("W")

    W = SpinGlassEngine.mpo(ctr, ctr.layers.main, i, indβ)

    println("rand")
    ψ = rand(QMps{Float64}, local_dims(W, :down), Dcut)
    canonise!(ψ, :right)
    canonise!(ψ, :left)

    ψ0 = rand(QMps{Float64}, local_dims(W, :up), Dcut)
    canonise!(ψ0, :right)
    canonise!(ψ0, :left)

    println("var")
    overlap, env = variational_compress!(ψ0, W, ψ,
                ctr.params.variational_tol, ctr.params.max_num_sweeps)

    println("zipper")
    ψ1 = zipper(W, ψ, method=:psvd_sparse, Dcut=Dcut, tol=tolS)
end

# println("pre-sweep")
# clear_memoize_cache()
bench()
#println("measuring ... ")
# clear_memoize_cache()
# @profile bench()

# pprof(flamegraph(); webhost = "localhost", webport = 57328)
