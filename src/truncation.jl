using Tullio

function findeigen_l!(P,A,B; maxiter_pow=100, tol_pow=1e-12, X=similar(P[1,1,:,:]), Pold=copy(P))
    q = size(P,1)
    da, db = size(A,4), size(B,4)

    for it in 1:maxiter_pow
        copy!(Pold,P)
        @tullio P[x,z,i,j] = Pold[x,z,i,k] * A[x,x,z,k,j]
        @tullio P[x,z,i,j] += B[x,x,z,k,i] * Pold[x,z,k,j]
        @tullio P[x,z,i,j] += Pold[x,zz,i,j] * (zz!=z)
        for x in 1:q, z in 1:q
            for y in 1:q
                y==x && continue
                @tullio X[i,j] = B[$x,$y,$z,k,i] * Pold[$y,$z,k,j]
                # @views mul!(X, B[x,y,z,:,:]', Pold[y,z,:,:])
                @tullio P[$x,$z,i,j] += X[i,k] * A[$x,$y,$z,k,j]
                # @views mul!(P[x,z,:,:], X, A[x,y,z,:,:], 1.0, 1.0)
            end
        end
        P ./= sum(abs, P)
        ϵ = sum((p-pold)^2 for (p,pold) in zip(P,Pold))
        # @tullio ϵ := (Pold[i] - P[i])^2
        if ϵ < tol_pow
            @show it
            return P
        end
    end
    @show maxiter_pow
    return P
end
function findeigen_r!(Q,A,B; maxiter_pow=100, tol_pow=1e-12, X=similar(Q[1,1,:,:]), Qold=copy(Q))
    q = size(Q,1)
    da, db = size(A,4), size(B,4)

    for it in 1:maxiter_pow
        copy!(Qold, Q)
        @tullio Q[x,z,i,j] = Qold[x,z,i,k] * A[x,x,z,j,k]
        @tullio Q[x,z,i,j] += B[x,x,z,i,k] * Qold[x,z,k,j]
        @tullio Q[x,z,i,j] += Qold[x,zz,i,j] * (zz!=z)
        for x in 1:q, z in 1:q
            for y in 1:q
                y==x && continue
                @tullio X[i,j] = B[$x,$y,$z,i,k] * Qold[$y,$z,k,j]
                # @views mul!(X, B[x,y,z,:,:], Qold[y,z,:,:])
                @tullio Q[$x,$z,i,j] += X[i,k] * A[$x,$y,$z,j,k]
                # @views mul!(Q[x,z,:,:], X, A[x,y,z,:,:]', 1.0, 1.0)
            end
        end
        Q ./= sum(abs, Q)
        ϵ = sum((q-qold)^2 for (q,qold) in zip(Q,Qold))
        # @tullio ϵ := (Qold[i] - Q[i])^2
        if ϵ < tol_pow
            @show it
            return Q
        end
    end
    @show maxiter_pow
    return Q
end

function ascent!(A,B; maxiters=[100], ηs=[1e-2], tols=[1e-12], maxiter_pow=1000, tol_pow=1e-12)
    q = size(A, 1)
    da, db = size(A,4), size(B,4)
    Anew = copy(A)

    Q = rand(q,q,db,da)
    P = rand(q,q,db,da)
    Q1 = rand(q,q,da,da)
    P1 = rand(q,q,da,da)
    Qold = similar(Q)
    Pold = similar(P)
    Q1old = similar(Q1)
    P1old = similar(P1)

    simQ = zeros(size(Q,3), size(Q,4))
    simQ1 = zeros(size(Q1,3), size(Q1,4))
    X = zeros(size(Q1,3), size(Q1,4))

    @show sizeof(Anew) + sizeof(Q) + sizeof(P) + sizeof(Q1) +
          sizeof(P1) + sizeof(Qold) + sizeof(Pold) + sizeof(Q1old) + sizeof(P1old) +
          sizeof(simQ) + sizeof(simQ1) + sizeof(X)

    for i in eachindex(ηs)
        η = ηs[i]
        maxiter = maxiters[i]
        tol = tols[i]
        for it in 1:maxiter
            findeigen_l!(P, A, B; maxiter_pow, tol_pow, X=simQ, Pold)
            findeigen_r!(Q, A, B; maxiter_pow, tol_pow, X=simQ, Qold)
            findeigen_l!(P1, A, A; maxiter_pow, tol_pow, X=simQ1, Pold=P1old)
            findeigen_r!(Q1, A, A; maxiter_pow, tol_pow, X=simQ1, Qold=Q1old)

            # n = sum(tr(P[x,z,:,:]'Q[x,z,:,:]) for x in 1:q, z in 1:q)
            @tullio n := P[x,z,i,j] * Q[x,z,i,j]
            P ./= n
            # n1 = -2 * sum(tr(P1[x,z,:,:]'Q1[x,z,:,:]) for x in 1:q, z in 1:q)
            @tullio n1 := P1[x,z,i,j] * Q1[x,z,i,j]
            P1 ./= -2n1

            for x in 1:q, y in 1:q, z in 1:q
                # Step proportional to η*parameter
                if x==y
                    # @views mul!(X, P[x,z,:,:]', Q[x,z,:,:])
                    @tullio X[i,j] = P[$x,$z,k,i] * Q[$x,$z,k,j]
                    # @views mul!(X, P1[x,z,:,:]', Q1[x,z,:,:], 2.0, 1.0)
                    @tullio X[i,j] += 2 * P1[$x,$z,k,i] * Q1[$x,$z,k,j]
                else
                    # @views mul!(simQ1, P1[x,z,:,:]', A[x,y,z,:,:])
                    @tullio simQ1[i,j] = P1[$x,$z,k,i] * A[$x,$y,$z,k,j]
                    # @views mul!(X, simQ1, Q1[y,z,:,:])
                    @tullio X[i,j] = 2 * simQ1[i,k] * Q1[$y,$z,k,j]

                    # @views mul!(simQ', P[x,z,:,:]', B[x,y,z,:,:])
                    @tullio simQ[j,i] = P[$x,$z,k,i] * B[$x,$y,$z,k,j]
                    # @views mul!(X, simQ', Q[y,z,:,:], 1.0, 2.0)
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