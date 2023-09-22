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

onGPU = true #false


function bench(instance::String)
    m = 16
    n = 16
    t = 4

    β = 0.5
    bond_dim = 8
    DE = 16.0
    δp = 1E-5*exp(-β * DE)
    num_states = 32
    println("creating factor graph" )
    @time begin
    ig = ising_graph(instance)

    cl_h = clustered_hamiltonian(
    ig,
    spectrum=full_spectrum, #brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule = zephyr_lattice((m, n, t))
    )
    end
    params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    Strategy = Zipper # SVDTruncate
    Sparsity = Sparse #Dense
    tran = all_lattice_transformations[1]
    Layout = GaugesEnergy
    Gauge = NoUpdate
    println("creating network and contractor")
    @time begin
        net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparsity}(m, n, cl_h, tran)
        ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)
        println("Memory lp = ", format_bytes.(measure_memory(net.lp)), " elements = ", length(net.lp))
    end

    @time begin
        W = SpinGlassEngine.mpo(ctr, ctr.layers.main, 8, 4)
    end
    # for st in W.sites
    #     println(st, "  ", size(W[st]), "  btm ", size.(W[st].bot), " top ",  size.(W[st].top), " ctr ",  size(W[st].ctr))
    # end

    # st = 8
    # DD = 8
    # LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
    # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
    # B  = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    # @time for ii in 1:10
    #     xx = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)
    # end

    st = 17 // 2
    # println(size(W[st].ctr.con), " size  ", size.(Ref(W[st].ctr.lp), W[st].ctr.projs), " length  ", length.(Ref(W[st].ctr.lp), W[st].ctr.projs))
    DD = 8

    # # # println(" ")
    # println(" PROJECT ")
    # LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
    # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
    # B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    # yy = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)
    # @time for _ in 1:10
    #     yy = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)
    # end
    # yy1 = SpinGlassTensors.project_ket_on_bra2(LE, B, W[st], RE)
    # @time for _ in 1:1
    #     yy1 = SpinGlassTensors.project_ket_on_bra2(LE, B, W[st], RE)
    # end
    # println("DIFFERENCE = ", maximum(abs.(yy - yy1)))

    # println("  RIGHT  ")
    # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
    # A = CuArray(rand(Float64, DD, DD, size(W[st], 2)))
    # B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))

    # yy = SpinGlassTensors.update_env_right(RE, A, W[st], B)
    # @time for _ in 1:10
    #     yy = SpinGlassTensors.update_env_right(RE, A, W[st], B)
    # end
    # yy1 = SpinGlassTensors.update_env_right2(RE, A, W[st], B)
    # @time for _ in 1:1
    #     yy1 = SpinGlassTensors.update_env_right2(RE, A, W[st], B)
    # end
    # println("DIFFERENCE = ", maximum(abs.(yy - yy1)))

    # println("  LEFT  ")
    # LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
    # A = CuArray(rand(Float64, DD, DD, size(W[st], 2)))
    # B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    # yy = SpinGlassTensors.update_env_left(LE, A, W[st], B)
    # @time for _ in 1:10
    #     yy = SpinGlassTensors.update_env_left(LE, A, W[st], B)
    # end

    # yy1 = SpinGlassTensors.update_env_left2(LE, A, W[st], B)
    # @time for _ in 1:1
    #     yy1 = SpinGlassTensors.update_env_left2(LE, A, W[st], B)
    # end
    # println("DIFFERENCE = ", maximum(abs.(yy - yy1)))

    # println("solving")
    # @time sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    # println("Result ", sol.energies)
    # println("Memory lp = ", format_bytes.(measure_memory(net.lp)), " elements = ", length(net.lp))

    println("  REDUCED ")
    RE = CuArray(rand(Float64, DD, size(W[st], 3)))
    A = 3
    B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
    yy = SpinGlassTensors.update_reduced_env_right(RE, A, W[st], B)
    @time for _ in 1:10
        yy = SpinGlassTensors.update_reduced_env_right(RE, A, W[st], B)
    end
    yy1 = SpinGlassTensors.update_reduced_env_right2(RE, A, W[st], B)
    @time for _ in 1:10
        yy1 = SpinGlassTensors.update_reduced_env_right2(RE, A, W[st], B)
    end
    println("DIFFERENCE = ", maximum(abs.(yy - yy1)))

end

instance = "$(@__DIR__)/../test/instances/zephyr_random/Z8/RAU/SpinGlass/001_sg.txt"
bench(instance)
# @profile bench(instance)

# pprof(flamegraph(); webhost = "localhost", webport = 57323)
