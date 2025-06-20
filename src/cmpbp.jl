const CMPS{F,N} = Array{F,N} where {F,N}

function iterate_infinite_regular!(A::CMPS{F,5}, factor, prob;
    maxiter=100, tol=1e-12, maxits_ascent=[100], tols_ascent=[1e-12], ηs=[0.01] , maxit_pow=1000, tol_pow=1e-12) where F<:Number
    q,d = size(A,1), size(A,4)
    size(A,2)==size(A,3)==q && size(A,5)==d || error("Wrong dimensions for A, got $(size(A))")

    E(x,y; q=2) = [i==x && j==y for i in 1:q, j in 1:q]

    Anew = rand(size(A)...)
    # Anew = 0.5*ones(size(A)...)
    AA = zeros(F, q,q,q,(q*d)^2,(q*d)^2)
    for it in 1:maxiter
        for x in 1:q, y in 1:q, z in 1:q
            AA[x,y,z,:,:] = sum(prob(factor, y,x,x₁ᵗ,x₂ᵗ,z)*kron(E(x₁ᵗ,x₁ᵗ), E(x₂ᵗ,x₂ᵗ), I(d), I(d)) for x₁ᵗ in 1:q, x₂ᵗ in 1:q)
            if x==y
                AA[x,y,z,:,:] .+= (sum(kron(I(q), E(x₂ᵗ,x₂ᵗ⁺¹;q), I(d), A[x₂ᵗ,x₂ᵗ⁺¹,x,:,:]) for x₂ᵗ in 1:q, x₂ᵗ⁺¹ in 1:q) +
                                sum(kron(E(x₁ᵗ,x₁ᵗ⁺¹; q), I(q), A[x₁ᵗ,x₁ᵗ⁺¹,x,:,:], I(d)) for x₁ᵗ in 1:q, x₁ᵗ⁺¹ in 1:q))
            end
        end
        # Anew .= 0.5
        Anew .= rand.()
        ascent!(Anew,AA; maxiters=maxits_ascent, ηs=ηs, tols=tols_ascent, maxiter_pow=maxit_pow, tol_pow=tol_pow)
        @show marginals(AA) marginals(Anew)
        if norm(Anew - A) < tol
            return it, Anew
        end
        A .= Anew
    end
    return maxiter, Anew, AA
end