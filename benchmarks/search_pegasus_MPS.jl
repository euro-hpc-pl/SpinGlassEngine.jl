using SpinGlassEngine

# Solve instance using MPS search
function bench(instance::String, num_states::Int=100)
    β = 3.0
    bond_dimension = 32
    variational_tol = 1E-8
    max_num_sweeps = 10

    @time sol = low_energy_spectrum(
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

bench("$(@__DIR__)/instances/pegasus_droplets/2_2_3_00.txt")
