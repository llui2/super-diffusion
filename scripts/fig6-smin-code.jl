include("scripts/SuperDiffusion.jl")

# ======== Initializations ================================

N = 500
k1 = 25
k2 = 50
MCS = 1e3
D = 2

# ======== Simulated Annealing ================================

perms = []
lambdas = []
sigmas = []
final_lambdas = []

# --- Free SA with ER

seed = rand(1:9999)
seed = 3601 # 40 489 # seed for results in the paper
Random.seed!(seed)
println("\n ER seed = ", seed)
G1, G2 = generate_graphs(N, k1, k2, "ER", "ER")

for i in 1:D÷2
    Temp = 10.0 # initial temperature for free SA (allow for more exploration)

    println("\u1b[1F")
    println(string("D = ", i, "\u1b[1F"))

    σ, δs, λs = SA(G1, G2, MCS, Temp, false)
    push!(perms, σ)
    push!(lambdas, λs)
    push!(sigmas, δs)
    push!(final_lambdas, λs[end])
end

# --- Free SA with SF

seed = rand(1:9999)
seed = 164 # 917 # seed for results in the paper
Random.seed!(seed)
println("\n SF seed = ", seed)

G1, G2 = generate_graphs(N, k1, k2, "SF", "SF")

for i in 1:D÷2
    Temp = 10.0 # initial temperature for free SA (allow for more exploration)

    println("\u1b[1F")
    println(string("D = ", D ÷ 2 + i, "\u1b[1F"))

    σ, δs, λs = SA(G1, G2, MCS, Temp, false)
    push!(perms, σ)
    push!(lambdas, λs)
    push!(sigmas, δs)
    push!(final_lambdas, λs[end])
end

#--------------------------------------------------------------------------------------

df = DataFrame(t=0:MCS)
for i in 1:2
    df[!, "λ$i"] = lambdas[i]
    df[!, "δ$i"] = sigmas[i]
end
CSV.write("smin.csv", df)

run(`python3 scritps/fig6-smin-plot.py`)