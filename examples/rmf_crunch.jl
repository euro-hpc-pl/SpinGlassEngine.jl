using JLD2
using SpinGlassEngine
using Images, Colors
using LinearAlgebra

sol_path = "$(@__DIR__)/bench_results/rmf"
sol = load_object(joinpath(sol_path, "strawbery_sol_2.jld2"))

states = sol.states[1]
states = reshape(states, (240, 320))



# Define a mapping from integers to colors
color_map = Dict(
    1 => RGB(1.0, 1.0, 1.0),   # white
    2 => RGB(212/255, 199/255, 190/255),  
    3 => RGB(176/255, 171/255, 130/255),  
    4 => RGB(85/255, 112/255, 48/255),
    5 => RGB(52/255, 76/255, 16/255), 
    6 => RGB(133/255, 140/255, 85/255),   
    7 => RGB(150/255, 75/255, 48/255),   
    8 => RGB(209/255, 115/255, 132/255), 
    9 => RGB(132/255, 19/255, 20/255),   
    10 => RGB(200/255, 74/255, 83/255), 
    11 => RGB(177/255, 13/255, 18/255), 
    12 => RGB(196/255, 36/255, 43/255)

)


# Get the dimensions of the matrix
rows, cols = size(states)

# Create an empty image
img = zeros(RGB, rows, cols)

# Assign colors to the image based on the matrix values
for i in 1:rows
    for j in 1:cols
        img[i, j] = color_map[states[i, j]]
    end
end
img = transpose(img)
# Display the image

save(joinpath("bench_results", "rmf", "sol_2.png"), img)