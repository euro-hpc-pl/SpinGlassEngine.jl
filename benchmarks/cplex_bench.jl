using CPLEX, JuMP
using SpinGlassNetworks, Graphs
using LinearAlgebra
using CSV, DataFrames


function qubo(h::Vector, J::Matrix)
    b = [sum(J[i, :]) + sum(J[:, i]) for i ∈ 1:length(h)]
    4 .* J .+ 2 .* Diagonal(h .- b)
end

function run_cplex(instance::String, out_dir::String)
    ig = ising_graph(instance)
    h = biases(ig)
    J = couplings(ig)
    Q = qubo(h, J)

    inst = instance[end-9:end-7] # works only for standard instance file format
    out_path = string(out_dir, "/", inst, ".csv")

    model = Model(CPLEX.Optimizer)
    #set_silent(model)
    @variable(model, x[i ∈ vertices(ig)], Bin)
    @objective(model, Min, x.data' * Q * x.data)
    optimize!(model)

    σ = value.(x.data)
    s = [i ≈ 1 ? 1 : -1 for i in σ]
    E_qubo = objective_value(model) + sum(J) - sum(h)
    E_ising = s' * J * s + s' * h

    data = DataFrame(
        :instance => inst,
        :state => [s],
        :energy_qubo => E_qubo,
        :energy_ising => E_ising,
        :time => round(solve_time(model), digits=2),
        :status => raw_status(model) 
        
    )
    println(data)
    CSV.write(out_path, data, delim = ';', append = false)

end


instance_dir = "$(@__DIR__)/../test/instances/zephyr_random/Z2/RAU/SpinGlass"
#instance = "$(@__DIR__)/../test/instances/pathological/pegasus_3_4_1.txt"
#instance = Dict((1, 2) => 1.0, (1, 3) => 1.0)

for file in readdir(instance_dir, join=true)
    run_cplex(file, "$(@__DIR__)/results/CPLEX/Z2/RAU/tmp")
end

# out =  "$(@__DIR__)/results/CPLEX/Z2/AC3/tmp"
# for i in collect(39:100)
#     inst_name = "00$i"[end-2: end] * "_sg.txt"
#     inst = instance_dir * "/" * inst_name
#     run_cplex(inst, out)
# end


