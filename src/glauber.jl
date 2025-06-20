function prob_glauber(factor, y,x,x∂i...; q=2)
    β, J, h = factor.β, factor.J, factor.h
    localfield = β*(h + J*sum(potts2spin, x∂i))
    if x!=y
        return exp(potts2spin(y)*localfield) / (2*cosh(localfield))
    else
        return -sum(exp(potts2spin(y)*localfield) / (2*cosh(localfield)) for y in 1:q if y≠x)
    end
end

struct GlauberFactor{F}
    β::F
    J::F
    h::F
end