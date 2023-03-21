using SpinGlassEngine
using CSV
using Tables
using DataFrames
using Statistics

num_states = 100
N = [8,]
bet = 0.1
betas = collect(0.5:0.5:4.0)
instances = collect(1:1:100)
tau_nn = zeros(length(N), length(betas))
tau_nnn = zeros(length(N), length(betas))
tau_plaq = zeros(length(N), length(betas))
rr = []
bd = 16
for (a,k) in enumerate(N) # wielkość układu
    for (j,b) in enumerate(betas) # betas
        csv = CSV.File("benchmarks/results/square_gauss/S$(k)/sampling_tau/merged.csv", delim = ";") #|> Tables.matrix
        # csv = CSV.File("benchmarks/results/square_gauss/S$(k)/beta_0_1/merged.csv", delim = ";") #|> Tables.matrix

        hlw_sol = zeros(Float64, length(instances))
        hlw_sol_nnn = zeros(Float64, length(instances))
        hlw_plaq = zeros(Float64, length(instances))
        hlw_r = []
        for (i,m) in enumerate(instances) # average over instances
            data = DataFrame(csv)
            if m in collect(1:1:9)
                data = data[data.instance .== "00$(m).txt", :]
                data = data[data.β .== b, :]
            elseif m in collect(10:1:99)
                data = data[data.instance .== "0$(m).txt", :]
                data = data[data.β .== b, :]
            else
                data = data[data.instance .== "$(m).txt", :]
                data = data[data.β .== b, :]
            end
            overlap = measure_overlap(data, k, num_states)
            hlw_sol[i] = mean(overlap.HLW_nn)
            hlw_sol_nnn[i] = mean(overlap.HLW_nnn)
            hlw_plaq[i] = mean(overlap.HLW_plaq)
            append!(hlw_r, overlap.taur)
        end
        hlw_r = permutedims(reshape(hlw_r, (Int(k/2), :)), (2, 1))
        push!(rr, mean(hlw_r, dims=1))

        tau_nn[a, j] = mean(hlw_sol)
        tau_nnn[a, j] = mean(hlw_sol_nnn)
        tau_plaq[a, j] = mean(hlw_plaq)
    end
end
r = reshape(rr, (length(betas), length(N)))

tau_nn = DataFrame(tau_nn, :auto)
CSV.write("tau_nn_sampling_tau2.csv", tau_nn)

tau_nnn = DataFrame(tau_nnn, :auto)
CSV.write("tau_nnn_sampling_tau2.csv", tau_nnn)

tau_plaq = DataFrame(tau_plaq, :auto)
CSV.write("tau_plaq_sampling_tau2.csv", tau_plaq)

tau_r = DataFrame(r, :auto)
CSV.write("tau_r_sampling_tau2.csv", tau_r)