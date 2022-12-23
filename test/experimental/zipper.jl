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
tran =  LatticeTransformation((1, 2, 3, 4), false)
Layout = EnergyGauges
Gauge = NoUpdate

i = div(m, 2)
indβ = 1

net = PEPSNetwork{Square{Layout}, Sparse}(m, n, fg, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)
Ws = SpinGlassEngine.mpo(ctr, ctr.layers.main, i, indβ)

net = PEPSNetwork{Square{Layout}, Dense}(m, n, fg, tran)
ctr = MpsContractor{Strategy, Gauge}(net, [β], :graduate_truncate, params)
Wd = SpinGlassEngine.mpo(ctr, ctr.layers.main, i, indβ)

Dcut = ctr.params.bond_dimension
tolV = ctr.params.variational_tol
tolS = ctr.params.tol_SVD
max_sweeps = ctr.params.max_num_sweeps

println("Dcut = ", Dcut, " tolV = ", tolV, " tolS = ", tolS, " max_sweeps = ", max_sweeps, " i = ", i)

ψ = IdentityQMps(Float64, local_dims(Wd, :down), ctr.params.bond_dimension) # F64 for now
canonise!(ψ, :left)

ψ0 = dot(Wd, ψ)
@time canonise_truncate!(ψ0, :right, Dcut, tolS)

@time ψ1 = zipper(Wd, ψ, Dcut, tolS)
@time ψ2 = zipper(Ws, ψ, Dcut, tolS)

println(dot(ψ0, ψ0))
println(dot(ψ1, ψ1))
println(dot(ψ2, ψ2))

println(dot(ψ0, ψ1) / (norm(ψ0) * norm(ψ1)))
println(dot(ψ0, ψ2) / (norm(ψ0) * norm(ψ2)))
println(dot(ψ1, ψ2) / (norm(ψ1) * norm(ψ2)))

println(" Wd -> ", format_bytes(measure_memory(Wd)))
println(" Ws -> ", format_bytes(measure_memory(Ws)))
println(" ψ -> ", format_bytes(measure_memory(ψ)))
println(" ψ0 -> ", format_bytes(measure_memory(ψ0)))