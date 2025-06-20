# include("../src/CMPBP.jl")
# using .CMPBP
include("../src/observables.jl")
include("../src/truncation.jl")
include("../src/glauber.jl")
include("../src/cmpbp.jl")
using LinearAlgebra, JLD2

q = 2
# d = 5
# A = rand(q,q,q,d,d)

k = 3
β = 1.0
J = 0.2
h = 0.1

w = GlauberFactor(β, J, h)

using MatrixProductBP.Models
g = Models.RandomRegular(3)
eq = equilibrium_observables(g, J; β, h)
m = eq.m
@show m

for d in 5:10
    @show d
    A = rand(q,q,q,d,d)
    it, A, AA = iterate_infinite_regular!(A, w, prob_glauber; maxiter=10, tol=1e-12, maxits_ascent=[300, 300, 300], tols_ascent=[1e-16, 1e-16, 1e-16], ηs=[1e-2, 1e-3, 1e-4], maxit_pow=10^3, tol_pow=1e-16)
    jldsave("simulations/results/infinite_3-regular_d$d.jld2"; AA, A)
    @show magnetization(A)
end