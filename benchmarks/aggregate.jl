using CSV
using DataFrames



function aggregate(type::String, category::String, solvers::Vector{String})
    df = "CPLEX" in solvers ? CSV.read("$(@__DIR__)/bench_results/pegasus_random/$(type)/$(category)_CPLEX.csv", DataFrame) : 
                              DataFrame(energy_qubo=[missing for _ in 1:100], time = [missing for _ in 1:100])

    df2 = "DWave" in solvers ? CSV.read("$(@__DIR__)/bench_results/pegasus_random/$(type)/$(category)_dwave.csv", DataFrame) :
                               DataFrame(best_dwave=[missing for _ in 1:100])
    df3 = "SB" in solvers ? CSV.read("$(@__DIR__)/bench_results/pegasus_random/$(type)/$(category)_SB.csv", DataFrame) :
                                DataFrame(energy=[missing for _ in 1:100], time=[missing for _ in 1:100])
    instances = collect(1:100)
    best_dwave = df2[!, :best_dwave]
    CPLEX = df[!, :energy_qubo]
    CPLEX_time = df[!, :time]
    SB = df3[!, :energy]
    SB_time = df3[!, :time]
    
    result = DataFrame(instance=instances, 
                       DWave=best_dwave, 
                       CPLEX=CPLEX, 
                       CPLEX_time=CPLEX_time,
                       SB=SB,
                       SB_time=SB_time)
                       
    result
end


df = aggregate("P4", "RAU", ["CPLEX", "SB"])
CSV.write("$(@__DIR__)/bench_results/aggregated/P4_RAU.csv", df)