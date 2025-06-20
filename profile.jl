using Pkg
Pkg.activate(".")

using Revise, JLD2, Plots, LinearAlgebra
includet("src/observables.jl")
includet("src/truncation.jl")
includet("src/glauber.jl")
includet("src/cmpbp.jl")

AAs = []
As = []

for d in 5:8
    D = load("simulations/results/infinite_3-regular_d$d.jld2")
    push!(AAs, D["AA"])
    push!(As, D["A"])
    println("d = $d")
    println("magnetization before truncation: $(magnetization(AAs[end]))")
    println("magnetization after truncation: $(magnetization(As[end]))")
end

d = 5

AA = copy(AAs[d-4])
A = rand(size(As[d-4])...)

ascent!(A, AA; maxiters=[10], ηs=[1e-2], tols=[1e-16], maxiter_pow=10^3, tol_pow=1e-16)
@show magnetization(AA) magnetization(A);

using Profile
Profile.clear()
@profview_allocs ascent!(A, AA; maxiters=[100], ηs=[1e-2], tols=[1e-16], maxiter_pow=10^3, tol_pow=1e-16)