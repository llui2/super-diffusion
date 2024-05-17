include("SuperDiffusion.jl")

# ======== Initializations ================================

seed = rand(1:999)
# seed = 668 # seed for results in the paper
Random.seed!(seed)
println("\n seed = ", seed)

N = 500
k1 = 25
k2 = 50
MCS = 1e6
D = 10

G1, G2 = generate_graphs(N, k1, k2, "ER", "ER")

sort1 = sortperm(degree(G1), rev=false)
G1 = induced_subgraph(G1, sort1)[1]

sort2 = sortperm(degree(G2), rev=true)
G2_0 = induced_subgraph(G2, sort2)[1]

# ======== Simulated Annealing ================================

perms = []
lambdas = []
sigmas = []
final_lambdas = []

# --- Free SA

for i in 1:D÷2
    Temp = 0.01 # initial temperature for free SA

    println("\u1b[1F")
    println(string("D = ", i, "\u1b[1F"))

    σ, δs, λs = SA(G1, G2_0, MCS, Temp, false)
    push!(perms, σ)
    push!(lambdas, λs)
    push!(sigmas, δs)
    push!(final_lambdas, λs[end])
end

# --- Restricted SA

for i in 1:D÷2
    Temp = 1.0 # initial temperature for free SA

    println("\u1b[1F")
    println(string("D = ", D ÷ 2 + i, "\u1b[1F"))

    σ, δs, λs = SA(G1, G2_0, MCS, Temp, true)
    push!(perms, σ)
    push!(lambdas, λs)
    push!(sigmas, δs)
    push!(final_lambdas, λs[end])
end

free_best = argmax(final_lambdas[1:D÷2])
restricted_best = argmax(final_lambdas[D÷2+1:end]) + D ÷ 2
annealing_info = DataFrame(free_best=free_best, restricted_best=restricted_best)
CSV.write("data/annealing-info.csv", annealing_info)

#--------------------------------------------------------------------------------------

df = DataFrame(t=0:MCS)
for i in 1:D
    df[!, "λ$i"] = lambdas[i]
    df[!, "δ$i"] = sigmas[i]
end
CSV.write("data/annealing.csv", df)