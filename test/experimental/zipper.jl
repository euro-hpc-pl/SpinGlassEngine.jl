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

m, n, t = 8, 8, 8

β = 1
bond_dim = 8

ig = ising_graph("$(@__DIR__)/../instances/chimera_droplets/512power/001.txt")

fg = factor_graph(
    ig,
    spectrum= brute_force_gpu, #rm _gpu to use CPU
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-16, 1)

Strategy = SVDTruncate
Sparsity = Dense
tran =  LatticeTransformation((1, 2, 3, 4), false)
Layout = EnergyGauges
Gauge = NoUpdate

net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, tran)
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

ψ0 = dot(W, ψ)
@time canonise_truncate!(ψ0, :right, Dcut, tolS)
@time ψ1 = zipper(W, ψ, Dcut, tolS)

println(dot(ψ0, ψ0))
println(dot(ψ1, ψ1))
println(dot(ψ0, ψ1) / (norm(ψ0) * norm(ψ1)))

println(format_bytes(measure_memory(W)))
println(format_bytes(measure_memory(ψ)))
println(format_bytes(measure_memory(ψ0)))