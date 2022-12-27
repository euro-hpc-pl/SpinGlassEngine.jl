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

m = 10
n = 10
t = 3

β = 1
bond_dim = 3

ig = ising_graph("$(@__DIR__)/../test/instances/zephyr_random/Z4/RAU/SpinGlass/001_sg.txt")

fg = factor_graph(
    ig,
    # max_cl_states,
    spectrum = full_spectrum,  #brute_force_gpu, # rm _gpu to use CPU
    cluster_assignment_rule = zephyr_lattice_5tuple_rotated(m+1, n+1, zephyr_lattice_5tuple((Int(m/2),Int(n/2),t)))
)

params = MpsParameters(bond_dim, 1E-16, 1)

Strategy = MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
tran =  LatticeTransformation((1, 2, 3, 4), false)
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
println("Mpo memory = ", format_bytes(measure_memory(W)))

# @time begin
#     println("Rand and canonise ")
#     ψ = rand(QMps{Float64}, local_dims(W, :down), Dcut)
#     canonise!(ψ, :right)
#     canonise!(ψ, :left)
#     println("Mps memory = ", format_bytes(measure_memory(ψ)))
# end

# ψ0 = rand(QMps{Float64}, local_dims(W, :up), Dcut)
# canonise!(ψ0, :right)
# canonise!(ψ0, :left)

# @time ψ1 = zipper(W, ψ, :psvd_sparse, Dcut=Dcut, tol=tolS)

# @time begin
#     println("Variational compress ")
#     overlap, env = variational_compress!(ψ0, W, ψ, 
#                 ctr.params.variational_tol, ctr.params.max_num_sweeps)
# end

# println("Psi memory = ", format_bytes(measure_memory(ψ0)))
# println("Psi memory = ", format_bytes(measure_memory(ψ1)))
# println("Env memory = ", format_bytes(measure_memory(env)))

# println(dot(ψ0, ψ0))
# println(dot(ψ1, ψ1))
# println(dot(ψ0, ψ1))
