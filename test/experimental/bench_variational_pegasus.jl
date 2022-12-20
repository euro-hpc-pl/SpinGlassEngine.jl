using SpinGlassExhaustive
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

function brute_force_gpu(ig::IsingGraph; num_states::Int)
     brute_force(ig, :GPU, num_states=num_states)
end

m = 7
n = 7
t = 3

β = 1
bond_dim = 2

ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P8/RAU/SpinGlass/001_sg.txt")

fg = factor_graph(
    ig,
    spectrum= brute_force_gpu, #rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-16, 1)

Strategy = MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
tran =  LatticeTransformation((3, 4, 1, 2), false)
Layout = EnergyGauges
Gauge = NoUpdate

net = PEPSNetwork{SquareStar2{Layout}, Sparsity}(m, n, fg, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)

Dcut = ctr.params.bond_dimension
tolV = ctr.params.variational_tol
tolS = ctr.params.tol_SVD
max_sweeps = ctr.params.max_num_sweeps

i = div(m, 2)
indβ = 1

println("Dcut = ", Dcut, " tolV = ", tolV, " tolS = ", tolS, " max_sweeps = ", max_sweeps, " i = ", i)

W = SpinGlassEngine.mpo(ctr, ctr.layers.main, i, indβ)
ψ = IdentityQMps(Float64, local_dims(W, :down), ctr.params.bond_dimension) # F64 for now
canonise!(ψ, :left)

ψ0 = IdentityQMps(Float64, local_dims(W, :up), ctr.params.bond_dimension) # F64 for now
canonise!(ψ0, :left)

@time begin
    overlap, env = variational_compress!(ψ0, W, ψ, 
                ctr.params.variational_tol, ctr.params.max_num_sweeps)
end

println(env)

clear_memoize_cache()
