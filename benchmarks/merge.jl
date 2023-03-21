using CSV
using DataFrames
using ProgressMeter

# OUT_FILE = "$(@__DIR__)/results/512power/merged.csv"
# CSV_DIR = "$(@__DIR__)/results/512power/tmp"

# OUT_FILE  = "$(@__DIR__)/results/1152power/merged.csv"
# CSV_DIR = "$(@__DIR__)/results/1152power/tmp"

folder = "RCO"
OUT_FILE  = "$(@__DIR__)/results/square_gauss/S8/beta1.1/merged.csv"
CSV_DIR = "$(@__DIR__)/results/square_gauss/S8/beta1.1"

function merge(csv_dir::String, out_path::String)

    data = DataFrame()
    @showprogress "Merging: " for csv ∈ readdir(csv_dir, join=true)
        row = DataFrame(CSV.File(csv, delim = ";"))
        data = vcat(data, row, cols=:union)
    end
    CSV.write(out_path, data, delim = ';')
end

merge(CSV_DIR, OUT_FILE)
