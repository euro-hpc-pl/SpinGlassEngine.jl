# spawn one worker per device
using Distributed, CUDA

n_gpus = length(devices())
addprocs(n_gpus)

@everywhere begin
   using CUDA, LinearAlgebra

    function assign_resources(device_id, n_cores)
        CUDA.device!(device_id)
        BLAS.set_num_threads(n_cores)
    end


end


asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        assign_resources(d, 4)
        c = BLAS.get_num_threads()
        @info "Worker $p uses device $d and $c  cores"
    end
end

@distributed for i ∈ collect(1:7)
    println(device())
    println(i)
end

rmprocs()
# After every use close julia, workers are left even so they shoud be deleted
