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
    println("creating factor graph" )

    @time begin
    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        spectrum=full_spectrum,  # brute_force_gpu, # rm _gpu to use CPU
        cluster_assignment_rule = pegasus_lattice((m, n, t))
    )
    end

    clear_memoize_cache()
    params = MpsParameters(bond_dim, 1E-2, 5, 1E-16)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    Strategy = Zipper # SVDTruncate
    Sparsity = Sparse #Dense
    Layout = GaugesEnergy
    Gauge = NoUpdate

    for tran ∈ all_lattice_transformations

        println("creating network and contractor")
        @time begin
            net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparsity}(m, n, cl_h, tran)
            ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params; onGPU=onGPU)
        end
        println("Memory lp = ", format_bytes.(measure_memory(net.lp)), " elements = ", length(net.lp))
        println("Memory memoize = ", measure_memory(Memoization.caches))

        @time begin
            W = SpinGlassEngine.mpo(ctr, ctr.layers.main, 4, 1)
        end
        # for st in W.sites
        #     println(st, "  ", size(W[st]), "  btm ", size.(W[st].bot), " top ",  size.(W[st].top), " ctr ",  size(W[st].ctr))
        # end

        # println("  PROJECT SITE ")

        # st = 4
        # DD = bond_dim
        # LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
        # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
        # B  = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
        # xx = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)

        # @time begin
        #     CUDA.@sync begin
        #         for ii in 1:30
        #             xx = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)
        #         end
        #     end
        # end

        st = 9 // 2
        # println(size(W[st].ctr.con), " size  ", size.(Ref(W[st].ctr.lp), W[st].ctr.projs), " length  ", length.(Ref(W[st].ctr.lp), W[st].ctr.projs))
        DD = bond_dim

        # println(" ")
        # println(" PROJECT ")
        # LE = CuArray(rand(Float64, DD, DD, size(W[st], 1)))
        # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
        # B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
        # yy = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)
        # @time for _ in 1:100
        #     yy = SpinGlassTensors.project_ket_on_bra(LE, B, W[st], RE)
        # end
        # yy1 = SpinGlassTensors.project_ket_on_bra2(LE, B, W[st], RE)
        # @time for _ in 1:100
        #     yy1 = SpinGlassTensors.project_ket_on_bra2(LE, B, W[st], RE)
        # end
        # println("DIFFERENCE = ", maximum(abs.(yy - yy1)))


        # println("  RIGHT  ")
        # RE = CuArray(rand(Float64, DD, DD, size(W[st], 3)))
        # A = CuArray(rand(Float64, DD, DD, size(W[st], 2)))
        # B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
        # yy = SpinGlassTensors.update_env_right(RE, A, W[st], B)
        # @time for _ in 1:100
        #     yy = SpinGlassTensors.update_env_right(RE, A, W[st], B)
        # end
        # yy1 = SpinGlassTensors.update_env_right2(RE, A, W[st], B)
        # @time for _ in 1:100
        #     yy1 = SpinGlassTensors.update_env_right2(RE, A, W[st], B)
        # end
        # println("DIFFERENCE = ", maximum(abs.(yy - yy1)))

        # println("  LEFT  ")
        # LE = CuArray(rand(Float64, DD-1, DD, size(W[st], 1)))
        # A = CuArray(rand(Float64, DD, DD+1, size(W[st], 2)))
        # B = CuArray(rand(Float64, DD-1, DD+2, size(W[st], 4)))
        # yy = SpinGlassTensors.update_env_left(LE, A, W[st], B)
        # @time for _ in 1:1
        #     yy = SpinGlassTensors.update_env_left(LE, A, W[st], B)
        # end
        # yy1 = SpinGlassTensors.update_env_left2(LE, A, W[st], B)
        # @time for _ in 1:1
        #     yy1 = SpinGlassTensors.update_env_left2(LE, A, W[st], B)
        # end
        # println("DIFFERENCE = ", maximum(abs.(yy - yy1)))

        println("  REDUCED ")
        RE = CuArray(rand(Float64, DD, size(W[st], 3)))
        A = 3
        B = CuArray(rand(Float64, DD, DD, size(W[st], 4)))
        yy = SpinGlassTensors.update_reduced_env_right(RE, A, W[st], B)
        @time for _ in 1:1000
            yy = SpinGlassTensors.update_reduced_env_right(RE, A, W[st], B)
        end
        yy1 = SpinGlassTensors.update_reduced_env_right2(RE, A, W[st], B)
        @time for _ in 1:1000
            yy1 = SpinGlassTensors.update_reduced_env_right(RE, A, W[st], B)
        end
        println("DIFFERENCE = ", maximum(abs.(yy - yy1)))


    end

    # println("solving ... ")
    # try
    #     sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    #     println("Memory lp = ", format_bytes.(measure_memory(net.lp)), " elements = ", length(net.lp))
    #     println("Memory memoize", measure_memory(Memoization.caches))
    #     println("Result ", sol.energies)
    # catch err
    #     println("Memory lp = ", format_bytes.(measure_memory(net.lp)), " elements = ", length(net.lp))
    #     println("Memory memoize", measure_memory(Memoization.caches))
    #     println(err.msg)
    # end
end

instance = "$(@__DIR__)/../test/instances/pegasus_random/P8/CBFM-P/SpinGlass/001_sg.txt"
bench(instance)
# @profile bench(instance)
# pprof(flamegraph(); webhost = "localhost", webport = 57320)
