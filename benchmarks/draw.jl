using JLD2
using SpinGlassPEPS
using Images, Colors


instance = "$(@__DIR__)/instances/strawberry-glass-2-small.h5"
solutions = "$(@__DIR__)/bench_results/rmf"

width = 320 
height = 240

sol = load_object(joinpath(solutions, "strawbery_sol.jld2"))

number_to_color = Dict(
    1 => colorant"white",
    2 => colorant"red",
    3 => colorant"blue",
    4 => colorant"green",
    5 => colorant"orange",
    6 => colorant"yellow",
    7 => colorant"cyan",
    8 => colorant"purple",
    9 => colorant"sienna",
    10 => colorant"black",
    11 => colorant"maroon2",
    12 => colorant"navyblue"
)
color_vector = sol.states[1]
color_vector = map(x -> number_to_color[x], color_vector)
color_matrix = reshape(color_vector, (width, height))
img = colorview(RGB, permutedims(color_matrix, (2, 1)))

using FileIO
save(joinpath(solutions, "output_image.png"), img)