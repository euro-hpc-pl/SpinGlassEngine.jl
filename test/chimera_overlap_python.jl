using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

ground_energy = -3336.773383

β = 0.5
bond_dim = 16
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

fg = factor_graph(
    ising_graph(instance),
    spectrum= full_spectrum, #my_brute_force,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)

Strategy = SVDTruncate
Gauge = NoUpdate
@testset "Compare the results for GaugesEnergy with Python" begin
    for Sparsity ∈ (Dense, Sparse) 
        network = PEPSNetwork{Square{GaugesEnergy}, Sparsity}(m, n, fg, rotation(0))
        ctr = MpsContractor{Strategy, Gauge}(network, [β/8, β/4, β/2, β], :graduate_truncate, params; onGPU=onGPU)
        @testset "Compare the results with Python" begin
            overlap_python = [0.2637787707674837, 0.2501621729619047, 0.2951954406837012]
            for i in 1:n-1
                psi_top = mps_top(ctr, i, 4)
                psi_bottom = mps(ctr, i+1, 4)
                overlap =  psi_top * psi_bottom
                @test isapprox(overlap, overlap_python[i], atol=1e-5)
            end
        end
        clear_memoize_cache()
    end
end

@testset "Compare the results for EnergyGauges with Python" begin
    overlap_python = [0.18603559878582027, 0.36463028391550056, 0.30532555472025247]
    for Sparsity ∈ (Dense, Sparse)
        net = PEPSNetwork{Square{EnergyGauges}, Sparsity}(m, n, fg, rotation(0))
        ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params; onGPU=onGPU)
        for i in 1:n-1
            psi_top = mps_top(ctr, i, 4)
            psi_bottom = mps(ctr, i+1, 4)
            overlap = psi_top * psi_bottom
            @test isapprox(overlap, overlap_python[i], atol=1e-5)
        end
        clear_memoize_cache()
    end
end
