using LinearAlgebra

function qubo(h::Vector, J::Matrix)
    b = [sum(J[i, :]) + sum(J[:, i]) for i ∈ 1:length(h)]
    4 .* J .+ 2 .* Diagonal(h .- b)
end

ising_energy(h::Vector, J::Matrix, s::Vector) = s' * J * s + dot(h, s)
qubo_energy(Q::Matrix, σ::Vector) = σ' * Q * σ
qubo_state(s::Vector) = (s .+ 1) ./ 2
offset(h::Vector, J::Matrix) = sum(J) - sum(h)

n = 100
T = Float64
J = triu(rand(T, n, n), 1)
h = rand(T, n)
ising_state = [rand() < 0.5 ? -1 : 1 for _ ∈ 1:n]

Q = qubo(h, J)
@assert ising_energy(h, J, ising_state) ≈ qubo_energy(Q, qubo_state(ising_state)) + offset(h, J)
