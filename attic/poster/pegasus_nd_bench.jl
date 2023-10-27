using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using SpinGlassExhaustive
using Logging

using DataFrames
using CSV

disable_logging(LogLevel(1))


function brute_force_gpu(ig::IsingGraph; num_states::Int)
    brute_force(ig, :GPU, num_states=num_states)
end


function bench(instance::String, beta::Float64)
    m = 7
    n = 7
    t = 3

    β = beta
    bond_dim = 4
    δp = 1e-10
    num_states = 128

    ig = ising_graph(instance)

    cl_h = clustered_hamiltonian(
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

    net = PEPSNetwork{PegasusSquare, Sparsity}(m, n, cl_h, tran)
    ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)


    @time sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    energy = sol.energies[begin]
    open("$(@__DIR__)/results_dwave/P8_2_b1.txt","a") do io
        println(io, "beta ", β)
        println(io, instance, "  ", energy)
     end
    
end

instance_dir = "$(@__DIR__)/instances/P8_2"
for instance in readdir(instance_dir, join=true)
        bench(instance, 0.5)
end

#bench("$(@__DIR__)/instances/P8/r_001_nd.txt", 2.0)


#=
instance = "$(@__DIR__)/instances/P4/r_007_nd.txt" #this one was worst

for beta ∈ betas
    bench(instance, beta)
end
=#