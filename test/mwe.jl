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
onGPU = false
transform = all_lattice_transformations[1]
ig = ising_graph(instance)
potts_h = potts_hamiltonian(
    ig,
    spectrum = full_spectrum,
    cluster_assignment_rule = pegasus_lattice((m, n, t)),
)

Gauge = NoUpdate
for T in [Float64, Float32]
    println("type ", T)
    energies = Vector{T}[]
    params = MpsParameters{T}(;
        bond_dim = bond_dim,
        var_tol = T(VAR_TOL),
        num_sweeps = MAX_SWEEPS,
        tol_SVD = T(TOL_SVD),
    )
    search_params = SearchParameters(; max_states = mstates, cutoff_prob = δp)

    net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity,T}(m, n, potts_h, transform)
    ctr = MpsContractor{Strategy,Gauge,T}(
        net,
        params;
        onGPU = onGPU,
        beta = T(β),
        graduate_truncation = true,
    )
    sol, s = low_energy_spectrum(ctr, search_params)
    @test eltype(sol.energies) == T
    println(sol.energies[1])
    clear_memoize_cache()
end
