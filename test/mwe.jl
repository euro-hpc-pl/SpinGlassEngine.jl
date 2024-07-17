using SpinGlassEngine, SpinGlassNetworks, SpinGlassTensors
using Test
m, n, t = 3, 4, 1

β = 0.5
bond_dim = 8
mstates = 10
instance = "$(@__DIR__)/instances/pathological/pegasus_3_4_1.txt"

iter = 2
δp = 0.0
MAX_SWEEPS = 0
VAR_TOL = 1E-16
TOL_SVD = 1E-16
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
METHOD = :psvd_sparse
Strategy = Zipper
Sparsity = Sparse
Layout = GaugesEnergy
onGPU = true
transform = all_lattice_transformations[1]
ig = ising_graph(instance)
cl_h = clustered_hamiltonian(
    ig,
    spectrum = full_spectrum,
    cluster_assignment_rule = pegasus_lattice((m, n, t)),
)

Gauge = NoUpdate
for T in [Float64, Float32]
    println("=========")
    println("type ", T)
    energies = Vector{T}[]
    params = MpsParameters{T}(;bond_dim, T(VAR_TOL), MAX_SWEEPS, T(TOL_SVD), ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
    search_params = SearchParameters(mstates, δp)

    net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity,T}(m, n, cl_h, transform)
    ctr = MpsContractor{Strategy,Gauge,T}(
        net,
        T[β/8, β/4, β/2, β],
        :graduate_truncate,
        params;
        onGPU = onGPU,
    )
    sol, s = low_energy_spectrum(ctr, search_params)
    @test eltype(sol.energies) == T
    println(sol.energies[1])
    clear_memoize_cache()
end