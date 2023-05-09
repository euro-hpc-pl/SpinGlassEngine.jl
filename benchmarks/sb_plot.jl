using Plots
using DataFrames, CSV

plotlyjs()

csv_dir = "$(@__DIR__)/bench_results/aggregated/P4_AC3.csv"
df = CSV.read(csv_dir, DataFrame)

instances = df.instance
dwave = df.DWave
sb = df.SB
plot(instances, [dwave, sb])
savefig("myplot.pdf") 