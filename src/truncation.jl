using Tullio

function dotprod(A,B)
    @tullio _ := A[x,z,i,j] * B[x,z,i,j]
end

function apply!(V,A,B; q=size(V,1), da=size(A,4), db=size(B,4), X=ones(db,da), Vold=copy(V))
    @tullio V[x,z,i,j] = Vold[x,z,i,k] * A[x,x,z,j,k]
    @tullio V[x,z,i,j] += B[x,x,z,i,k] * Vold[x,z,k,j]
    @tullio V[x,z,i,j] += Vold[x,zz,i,j] * (zz!=z)
    for x in 1:q, z in 1:q
        for y in 1:q
            y==x && continue
            @tullio X[i,j] = B[$x,$y,$z,i,k] * Vold[$y,$z,k,j]
            @tullio V[$x,$z,i,j] += X[i,k] * A[$x,$y,$z,j,k]
        end
    end
    return V
end

function apply_dag!(V,A,B; q=size(V,1), da=size(A,4), db=size(B,4), X=ones(db,da), Vold=copy(V))
    @tullio V[x,z,i,j] = Vold[x,z,i,k] * A[x,x,z,k,j]
    @tullio V[x,z,i,j] += B[x,x,z,k,i] * Vold[x,z,k,j]
    @tullio V[x,z,i,j] += Vold[x,zz,i,j] * (zz!=z)
    for x in 1:q, z in 1:q
        for y in 1:q
            y==x && continue
            @tullio X[i,j] = B[$y,$x,$z,k,i] * Vold[$y,$z,k,j]
            @tullio V[$x,$z,i,j] += X[i,k] * A[$y,$x,$z,k,j]
        end
    end
    return V
end

function findeigen_r!(Q,A,B; maxiter_pow=100, tol_pow=1e-12,
    q=size(V,1), da=size(A,4), db=size(B,4), X=ones(db,da), Qold=similar(Q))
    for it in 1:maxiter_pow
        copy!(Qold, Q)
        apply!(Q,A,B; q,da,db, X, Vold=Qold)
        Q ./= sum(abs, Q)
        ϵ = sum((q-qold)^2 for (q,qold) in zip(Q,Qold))
        if sqrt(ϵ) < tol_pow
            # @show it
            return Q
        end
    end
    @show maxiter_pow
    return Q
end

function findeigen_l!(P,A,B; maxiter_pow=100, tol_pow=1e-12,
    q=size(V,1), da=size(A,4), db=size(B,4), X=ones(db,da), Pold=similar(P))
    for it in 1:maxiter_pow
        copy!(Pold,P)
        apply_dag!(P,A,B; q,da,db, X, Vold=Pold)
        P ./= sum(abs, P)
        ϵ = sum((p-pold)^2 for (p,pold) in zip(P,Pold))
        if sqrt(ϵ) < tol_pow
            # @show it
            return P
        end
    end
    @show maxiter_pow
    return P
end

mutable struct Truncator{T}
    Q::Array{T,4}
    P::Array{T,4}
    λ::T
    Q1::Array{T,4}
    P1::Array{T,4}
    λ1::T
    Qold::Array{T,4}
    Pold::Array{T,4}
    Q1old::Array{T,4}
    P1old::Array{T,4}
    simQ::Array{T,2}
    simQ1::Array{T,2}
    X::Array{T,2}
    Y::Array{T,4}
end

function Truncator(::Type{T}, A, B) where T
    q = size(A, 1)
    da, db = size(A, 4), size(B, 4)

    Q = rand(T,q,q,db,da)
    P = rand(T,q,q,db,da)
    Q1 = rand(T,q,q,da,da)
    P1 = rand(T,q,q,da,da)
    Qold = similar(Q)
    Pold = similar(P)
    Q1old = similar(Q1)
    P1old = similar(P1)

    simQ = zeros(T, db,da)
    simQ1 = zeros(T, da,da)
    X = zeros(T, da,da)
    Y = similar(Q)

    return Truncator{T}(Q,P,one(T),Q1,P1,one(T),Qold,Pold,Q1old,P1old,simQ,simQ1,X,Y)
end

Truncator(A,B) = Truncator(Float64, A, B)


function ascent!(A, B, t::Truncator=Truncator(A,B);
    maxiters=[100], ηs=[1e-2], tols=[1e-12], maxiter_pow=1000, tol_pow=1e-12,
    q=size(A, 1), da=size(A, 4), db=size(B, 4))
    Anew = copy(A)
    Q, P, Q1, P1 = t.Q, t.P, t.Q1, t.P1
    Qold, Pold, Q1old, P1old = t.Qold, t.Pold, t.Q1old, t.P1old
    simQ, simQ1, X = t.simQ, t.simQ1, t.X

    for i in eachindex(ηs)
        η = ηs[i]
        maxiter = maxiters[i]
        tol = tols[i]
        for it in 1:maxiter
            findeigen_l!(P, A, B; maxiter_pow, tol_pow, q,da,db, X=simQ, Pold)
            findeigen_r!(Q, A, B; maxiter_pow, tol_pow, q,da,db, X=simQ, Qold)
            findeigen_l!(P1, A, A; maxiter_pow, tol_pow, q,da,db=da, X=simQ1, Pold=P1old)
            findeigen_r!(Q1, A, A; maxiter_pow, tol_pow, q,da,db=da, X=simQ1, Qold=Q1old)

            n = dotprod(P, Q)
            P ./= n
            n1 = dotprod(P1, Q1)
            P1 ./= -2n1

            for x in 1:q, y in 1:q, z in 1:q
                # Step proportional to η*parameter
                if x==y
                    @tullio X[i,j] = P[$x,$z,k,i] * Q[$x,$z,k,j]
                    @tullio X[i,j] += 2 * P1[$x,$z,k,i] * Q1[$x,$z,k,j]
                else
                    @tullio simQ1[i,j] = P1[$x,$z,k,i] * A[$x,$y,$z,k,j]
                    @tullio X[i,j] = 2 * simQ1[i,k] * Q1[$y,$z,k,j]

                    @tullio simQ[j,i] = P[$x,$z,k,i] * B[$x,$y,$z,k,j]
                    @tullio X[i,j] += simQ[k,j] * Q[$y,$z,k,i]
                end

                Anew[x,y,z,:,:] .+= @views η .* sign.(X) .* A[x,y,z,:,:]

                #Step proportional to η*gradient
                # if x==y
                #     Anew[x,x,z,:,:] .+= @views η .* (P[x,z,:,:]'Q[x,z,:,:] + 2 * P1[x,z,:,:]'Q1[x,z,:,:])
                # else
                #     Anew[x,y,z,:,:] .+= @views η .* (P[x,z,:,:]'B[x,y,z,:,:]*Q[y,z,:,:] + 2 * P1[x,z,:,:]'A[x,y,z,:,:]*Q1[y,z,:,:])
                # end
            end
            A .= Anew
        end
    end
end

function ascent(A,B; maxiter=100, η=0.1, tol=1e-12, maxiter_pow=1000, tol_pow=1e-12)
    X = copy(A)
    ascent!(X,B; maxiter, η, tol, maxiter_pow, tol_pow)
    return X
end


function aug_lagrange!(A,B, t::Truncator=Truncator(A,B);
    μs=[1e1], maxiters=[100], ηs=[1e-2], tols=[1e-12], maxiter_pow=1000, tol_pow=1e-12,
    q=size(A, 1), da=size(A, 4), db=size(B, 4))
    Q, P, λ, Q1, P1, λ1 = t.Q, t.P, t.λ, t.Q1, t.P1, t.λ1
    SQ, SP, S1Q1, S1P1 = t.Qold, t.Pold, t.Q1old, t.P1old
    simQ, simQ1, X, Y = t.simQ, t.simQ1, t.X, t.Y

    derλ = 0.0
    derλ1 = 0.0
    derQ = similar(Q)
    derP = similar(P)
    derQ1 = similar(Q1)
    derP1 = similar(P1)
    derA = similar(A)

    for μ in μs
        for i in eachindex(ηs)
            η = ηs[i]
            maxiter = maxiters[i]
            tol = tols[i]

            for it in 1:maxiter
                apply!(SQ,A,B; q,da,db, X=simQ, Vold=Q)
                @tullio SQ[x,z,i,j] += -λ * Q[x,z,i,j]
                apply_dag!(SP,A,B; q,da,db, X=simQ, Vold=P)
                @tullio SP[x,z,i,j] += -λ * P[x,z,i,j]
                apply!(S1Q1,A,A; q,da,db=da, X=simQ1, Vold=Q1)
                @tullio S1Q1[x,z,i,j] += -λ1 * Q1[x,z,i,j]
                apply_dag!(S1P1,A,A; q,da,db=da, X=simQ1, Vold=P1)
                @tullio S1P1[x,z,i,j] += -λ1 * P1[x,z,i,j]

                # λ
                derλ = 1 - dotprod(P,Q) - μ * dotprod(Q,SQ)

                # λ1
                derλ1 = -0.5 - dotprod(P1,Q1) - μ * dotprod(Q1,S1Q1)

                # Q
                apply_dag!(derP, A,B; q,da,db, X=simQ, Vold=SQ)
                @tullio derP[x,z,i,j] += -λ * SQ[x,z,i,j]
                @tullio derQ[x,z,i,j] = μ * derP[x,z,i,j] + SP[x,z,i,j]

                # P
                @tullio derP[x,z,i,j] = SQ[x,z,i,j]

                # Q1
                apply_dag!(derP1, A,A; q,da,db=da, X=simQ1, Vold=S1Q1)
                @tullio derP1[x,z,i,j] += -λ1 * S1Q1[x,z,i,j]
                @tullio derQ1[x,z,i,j] = μ * derP1[x,z,i,j] + S1P1[x,z,i,j]
                
                # P1
                @tullio derP1[x,z,i,j] = S1Q1[x,z,i,j]
                
                # A
                for x in 1:q, y in 1:q, z in 1:q
                    if x==y
                        # A(x,x,z)
                        @tullio derA[$x,$x,$z,i,j] = (P[$x,$z,k,i] + μ * SQ[$x,$z,k,i]) * Q[$x,$z,k,j]
                        @tullio derA[$x,$x,$z,i,j] += (P1[$x,$z,k,i] + μ * S1Q1[$x,$z,k,i]) * Q1[$x,$z,k,j]
                        @tullio derA[$x,$x,$z,i,j] += (P1[$x,$z,i,k] + μ * S1Q1[$x,$z,i,k]) * Q1[$x,$z,j,k]
                    else
                        # A(x,y,z)
                        @tullio simQ[i,j] = B[$x,$y,$z,i,k] * Q[$y,$z,k,j]
                        @tullio simQ1[i,j] = A[$x,$y,$z,i,k] * Q1[$y,$z,k,j]

                        @tullio derA[$x,$y,$z,i,j] = (P[$x,$z,k,i] + μ * SQ[$x,$z,k,i]) * simQ[k,j]
                        @tullio derA[$x,$y,$z,i,j] += (P1[$x,$z,k,i] + μ * S1Q1[$x,$z,k,i]) * simQ1[k,j]

                        @tullio simQ1[i,j] = A[$x,$y,$z,i,k] * Q1[$y,$z,j,k]
                        @tullio derA[$x,$y,$z,i,j] += (P1[$x,$z,i,k] + μ * S1Q1[$x,$z,i,k]) * simQ1[k,j]
                    end
                end
                # return derλ, derλ1, derQ, derP, derQ1, derP1, derA

                println("∂/∂λ")
                println(norm(derλ))
                println("∂/∂λ1")
                println(norm(derλ1))
                println("∇Q")
                println(maximum(derQ))
                println("∇P")
                println(maximum(derP))
                println("∇Q1")
                println(maximum(derQ1))
                println("∇P1")
                println(maximum(derP1))
                println("∇A")
                println(maximum(derA))
                
                # Step proportional to gradient
                λ += η * derλ
                λ1 += η * derλ1
                @tullio Q[x,z,i,j] += η * derQ[x,z,i,j]
                @tullio P[x,z,i,j] += η * derP[x,z,i,j]
                @tullio Q1[x,z,i,j] += η * derQ1[x,z,i,j]
                @tullio P1[x,z,i,j] += η * derP1[x,z,i,j]
                @tullio A[x,y,z,i,j] += η * derA[x,y,z,i,j]

                # Step proportional to parameter
                # λ *= (1 + η * sign(derλ))
                # λ1 *= (1 + η * sign(derλ1))
                # @tullio Q[x,z,i,j] *= (1 + η * sign(derQ[x,z,i,j]))
                # @tullio P[x,z,i,j] *= (1 + η * sign(derP[x,z,i,j]))
                # @tullio Q1[x,z,i,j] *= (1 + η * sign(derQ1[x,z,i,j]))
                # @tullio P1[x,z,i,j] *= (1 + η * sign(derP1[x,z,i,j]))
                # @tullio A[x,y,z,i,j] *= (1 + η * sign(derA[x,y,z,i,j]))

                t.λ = λ
                t.λ1 = λ1

                println()
                println("marginals(B) = $(marginals(B))")
                println("marginals(A) = $(marginals(A))")
                println("fidelity_eig(A,B) = $(fidelity_eig(A,B))")
            end
        end
    end
    return derλ, derλ1, derQ, derP, derQ1, derP1, derA
end
