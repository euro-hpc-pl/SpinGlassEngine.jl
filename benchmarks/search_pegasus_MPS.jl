using SpinGlassEngine
using SpinGlassNetworks

# Solve instance using MPS search
function bench(instance::String, num_states::Int=10000)
    β = 4.0
    bond_dimension = 64
    variational_tol = 1E-10
    max_num_sweeps = 10

    @time sol, s = low_energy_spectrum(
        ising_graph(instance),
        bond_dimension,
        variational_tol,
        max_num_sweeps,
        β/8.0,
        β,
        :lin,
        num_states
    )
    println("ground from MPS: ", sol.energies[begin])
end

# best energy found: -56.96875
bench("$(@__DIR__)/../test/instances/pegasus_droplets/2_2_3_00.txt")
