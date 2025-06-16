module CMPBP

using LinearAlgebra
using Statistics

export potts2spin, propagator, marginals, magnetization, ascent!, ascent

include("observables.jl")
include("truncation.jl")

end