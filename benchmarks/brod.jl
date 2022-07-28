
β = 1.0

n = 10^8
x = rand(n)

@time a = exp.(-β .* (x .- minimum(x)))

@time begin
    μ = minimum(x)
    b = exp.(-β .* (x .- μ))
end

@assert a ≈ b
