using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t - 0)

ground_energy = -3336.773383

β = 0.5
bond_dim = 16
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

cl_h = clustered_hamiltonian(
    ising_graph(instance),
    spectrum = full_spectrum, #my_brute_force,
    cluster_assignment_rule = super_square_lattice((m, n, t)),
)

params = MpsParameters{Float64}(; bd = bond_dim, ϵ = 1E-8, sw = 4)
search_params = SearchParameters(; max_states = num_states, cut_off_prob = δp)

Strategy = SVDTruncate
Gauge = NoUpdate
@testset "Compare the results for GaugesEnergy with Python (Sparsity = $Sparsity)" for Sparsity ∈
                                                                                       (
    Dense,
    Sparse,
)
    network = PEPSNetwork{SquareSingleNode{GaugesEnergy},Sparsity,Float64}(
        m,
        n,
        cl_h,
        rotation(0),
    )
    ctr = MpsContractor{Strategy,Gauge,Float64}(
        network,
        params;
        onGPU = onGPU,
        βs = [β / 8, β / 4, β / 2, β],
        graduate_truncation = :graduate_truncate,
    )
    @testset "Compare the results with Python" begin
        overlap_python = [0.2637787707674837, 0.2501621729619047, 0.2951954406837012]
        for i = 1:n-1
            psi_top = mps_top(ctr, i, 4)
            psi_bottom = mps(ctr, i + 1, 4)
            overlap = psi_top * psi_bottom
            @test isapprox(overlap, overlap_python[i], atol = 1e-5)
        end
    end
    clear_memoize_cache()
end

@testset "Compare the results for EnergyGauges with Python (Sparsity = $Sparsity)" for Sparsity ∈
                                                                                       (
    Dense,
    Sparse,
)
    overlap_python = [0.18603559878582027, 0.36463028391550056, 0.30532555472025247]
    net = PEPSNetwork{SquareSingleNode{EnergyGauges},Sparsity,Float64}(
        m,
        n,
        cl_h,
        rotation(0),
    )
    ctr = MpsContractor{Strategy,Gauge,Float64}(
        net,
        params;
        onGPU = onGPU,
        βs = [β / 8, β / 4, β / 2, β],
        graduate_truncation = :graduate_truncate,
    )
    for i = 1:n-1
        psi_top = mps_top(ctr, i, 4)
        psi_bottom = mps(ctr, i + 1, 4)
        overlap = psi_top * psi_bottom
        @test isapprox(overlap, overlap_python[i], atol = 1e-5)
    end
    clear_memoize_cache()
end
