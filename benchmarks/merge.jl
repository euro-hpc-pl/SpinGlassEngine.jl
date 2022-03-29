using CSV
using DataFrames

#INSTANCE_DIR = "$(@__DIR__)/instances/chimera_droplets/512power"
INSTANCE_DIR = "$(@__DIR__)/instances/chimera_droplets/1152power"

#OUTPUT_DIR = "$(@__DIR__)/results/512power/tmp"
OUTPUT_DIR = "$(@__DIR__)/results/1152power/tmp"

function merge(csv_dir::String, out_path::String)

    data = DataFrame()
    for csv ∈ readdir(csv_dir, join=true)
        row = DataFrame(CSV.File(csv, delim = ";"))
        data = vcat(data, row, cols=:union)
    end
    CSV.write(out_path, data, delim = ';')
end

merge(INSTANCE_DIR, OUTPUT_DIR)
