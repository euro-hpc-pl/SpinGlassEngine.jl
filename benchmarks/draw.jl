using JLD2
using SpinGlassEngine
using Images, Colors


instance = "$(@__DIR__)/instances/strawberry-glass-2-small.h5"
solutions = "$(@__DIR__)/bench_results/rmf"

width = 240 
height = 320

sol = load_object(joinpath(solutions, "strawbery_sol.jld2"))
println("SpinGlassPEPS_energy: ", sol.energies[1])
println("best_found: ", 11725.19)
number_to_color = Dict(
    1 => colorant"white",
    2 => colorant"rgb(212,199,190)",
    3 => colorant"rgb(176,171,130)",
    4 => colorant"rgb(85,112,48)",
    5 => colorant"rgb(52,76,16)",
    6 => colorant"rgb(133,140,85)",
    7 => colorant"rgb(150,75,48)",
    8 => colorant"rgb(209,115,132)",
    9 => colorant"rgb(132,19,20)",
    10 => colorant"rgb(200,74,83)",
    11 => colorant"rgb(177,13,18)",
    12 => colorant"rgb(196,36,43)"#"rgb(200,74,83)"
)
color_vector = sol.states[1]
color_vector = map(x -> number_to_color[x], color_vector)
color_matrix = reshape(color_vector, (width, height))
img = colorview(RGB, permutedims(color_matrix, (2, 1)))

using FileIO
save(joinpath(solutions, "output_image.png"), img)