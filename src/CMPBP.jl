module CMPBP

using LinearAlgebra
using Statistics

export propagator, marginals, ascent!, ascent

include("marginals.jl")
include("truncation.jl")

end