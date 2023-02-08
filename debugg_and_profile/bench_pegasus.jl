using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using SpinGlassExhaustive
using Logging
using Profile, PProf
using FlameGraphs
using CUDA

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
    bond_dim = 2
    DE = 16.0
    δp = 1E-5 * exp(-β * DE)
    num_states = 32
    println("creating factor graph" )
    @time begin
    ig = ising_graph(instance)

    fg, lp = factor_graph(
    ig,
    spectrum=full_spectrum, #brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule = pegasus_lattice((m, n, t))
    )
    end

    println("Memory lp = ", format_bytes.(measure_memory(lp)), " elements = ", length(lp))

    params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    Strategy = Zipper # SVDTruncate
    Sparsity = Sparse #Dense
    tran = all_lattice_transformations[3]
    Layout = GaugesEnergy
    Gauge = NoUpdate
    println("creating network and contractor")
    @time begin
        net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, lp, tran)
        ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)
    end
    @time begin
        W = SpinGlassEngine.mpo(ctr, ctr.layers.main, 4, 4)
    end
    for st in W.sites
        println(st, "  ", size(W[st]), "  btm ", size.(W[st].bot), " top ",  size.(W[st].top), " ctr ",  size(W[st].ctr))

        # println() <: VirtualTensor
        #     println(size(W[st].con))
        # end
    end

    # st = 4
    # DD = 8
    # LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
    # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
    # B  = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    # @time xx = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)

    st = 9 // 2
    println(size(W[st].ctr.con), " size  ", size.(Ref(W[st].ctr.lp), W[st].ctr.projs), " length  ", length.(Ref(W[st].ctr.lp), W[st].ctr.projs))
    DD = 4

    println("  XXXXXXXXXXXXXXXXXXXX  ")

    LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
    RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
    B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    @time yy = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)

    # println("  XXXXXXXXXXXXXXXXXXXX  ")

    # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
    # A = CuArray(rand(Float64, DD, DD, size(W[st], 2)))
    # B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    # @time yy = SpinGlassTensors.update_env_right(RE, A, W[st], B)

    # println("  XXXXXXXXXXXXXXXXXXXX  ")

    # LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
    # A = CuArray(rand(Float64, DD, DD, size(W[st], 2)))
    # B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    # @time yy = SpinGlassTensors.update_env_left(LE, A, W[st], B)

   # println("solving")
   # @time sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
   # println("Result ", sol.energies)
   println("Memory lp = ", format_bytes.(measure_memory(lp)), " elements = ", length(lp))
end

instance = "$(@__DIR__)/../test/instances/pegasus_random/P8/RAU/SpinGlass/001_sg.txt"
bench(instance)
#@profile bench(instance)

#pprof(flamegraph(); webhost = "localhost", webport = 57325)
