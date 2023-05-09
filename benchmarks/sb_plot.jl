using Plots
using DataFrames, CSV
using LaTeXStrings

plotlyjs()


dwave_csv_dir = "$(@__DIR__)/bench_results/pegasus_random/P4/AC3_dwave.csv"
dwave_df = CSV.read(dwave_csv_dir, DataFrame)

sb_csv_dir = "$(@__DIR__)/bench_results/pegasus_random/P4/AC3_SB.csv"
sb_df = CSV.read(sb_csv_dir, DataFrame)

instances = dwave_df.instance
dwave = dwave_df.best_dwave
sb = sb_df.energy
diff = [(sb[i] - dwave[i])/dwave[i] for i in collect(1:length(sb))]
plot(
    instances, diff, 
    seriestype = :bar, 
    title="Dwave vs SBM",
    xlabel = "instance number",
    ylabel = "normalized difference",
    legend=false,
    )

file_title = "sb_dwave_plot"
savefig("$(file_title).pdf") 