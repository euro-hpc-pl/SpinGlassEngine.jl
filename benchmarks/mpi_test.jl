using MPI

MPI.Init()
size = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

K = 20

for i ∈ (1+rank):size:K
    println("$i --> $rank")
end
