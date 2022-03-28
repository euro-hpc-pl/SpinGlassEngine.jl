using CSV
using DataFrames

function merge(csv_dir::String, out_path::String)

    data = DataFrame()
    for csv ∈ readdir(csv_dir, join=true)
        row = DataFrame(CSV.File(csv, delim = ";"))
        data = vcat(data, row, cols=:union)
    end
    CSV.write(out_path, data, delim = ';')
end

merge(
    "$(@__DIR__)/results/512power/tmp",
    "$(@__DIR__)/results/512power/merged.csv"
)
