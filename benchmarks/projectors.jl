using MPI
using LinearAlgebra
using MKL
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassExhaustive
using Logging
using CSV
using DataFrames
using Memoization

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end

m = 7
n = 7
t = 3 #3 for pegasus, 4 for zephyr

β = 1
bond_dim = 5
δp = 1e-10
num_states = 128

ig = ising_graph("$(@__DIR__)/../test/instances/pegasus_random/P8/RCO/SpinGlass/001_sg.txt")
#ig = ising_graph("$(@__DIR__)/../test/instances/zephyr_random/Z2/RCO/SpinGlass/001_sg.txt")

cl_h = clustered_hamiltonian(
    ig,
    spectrum= full_spectrum, #brute_force_gpu, #rm _gpu to use CPU
    #spectrum= full_spectrum, #for zephyr
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 2)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = Zipper # MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
tran =  rotation(0)
Layout = EnergyGauges
Gauge = NoUpdate
indβ = 3

net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparsity}(m, n, cl_h, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β/4, β/2, β], :graduate_truncate, params)
P = Set()
S = Set()
RS = Set()
LS = Set()
US = Set()
DS = Set()

for i in 1:m
    count_site = 0
    count_virtual = 0
    println("--------i--------- ", i)
    M = SpinGlassEngine.mpo(ctr, ctr.layers.main, i, indβ)
    for (s, mpo_tensor) in M.tensors
        ctrten = mpo_tensor.ctr
        #println("ctr ", typeof(ctrten))
        if typeof(ctrten) <: SiteTensor
            println("====")
            count_site = count_site + 4
            a, b, c, d = ctrten.projs
            push!(P, a, b, c, d)
            push!(S, a, b, c, d)
            push!(LS, a)
            push!(US, b)
            push!(RS, c)
            push!(DS, d)

            println("site ", length(S))
            println("site left ", length(LS))
            println("site up ", length(US))
            println("site right ", length(RS))
            println("site down ", length(DS))

        elseif typeof(ctrten) <: VirtualTensor
            count_virtual = count_virtual + 6
            a, b, c, d, e, f = ctrten.projs
            push!(P, a, b, c, d, e, f)
        end
    end
    println("count site ", count_site)
    println("count virtual ", count_virtual)

end
println("===========")
println("length all ", length(P))
println("length site ", length(S))
println("===========")


for (i, p) in enumerate(P) 
    println(i, " ", format_bytes.(measure_memory(p)))
end
#println("memory ", measure_memory.(P))