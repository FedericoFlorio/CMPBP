function findeigen_l!(P,A,B; maxiter_pow=100, tol_pow=1e-12, X=similar(P[1,1,:,:]))
    Pold = copy(P)
    q = size(P,1)
    da, db = size(A,4), size(B,4)

    for it in 1:maxiter_pow
        Pold .= P
        for x in 1:q, z in 1:q
            mul!(P[x,z,:,:], Pold[x,z,:,:], A[x,x,z,:,:])
            mul!(P[x,z,:,:], B[x,x,z,:,:]', Pold[x,z,:,:], 1.0, 1.0)
            for zz in 1:q
                zz==z && continue
                P[x,z,:,:] .+= Pold[x,zz,:,:]
            end
            for y in 1:q
                y==x && continue
                mul!(X, B[x,y,z,:,:]', Pold[y,z,:,:])
                mul!(P[x,z,:,:], X, A[x,y,z,:,:], 1.0, 1.0)
            end
        end
        P ./= sum(abs, P)
        if sum(abs2, P - Pold) < tol_pow
            # @show it
            return P
        end
    end
    @show maxiter_pow
    return P
end
function findeigen_r!(Q,A,B; maxiter_pow=100, tol_pow=1e-12, X=similar(Q[1,1,:,:]))
    Qold = copy(Q)
    q = size(Q,1)
    da, db = size(A,4), size(B,4)

    for it in 1:maxiter_pow
        Qold .= Q
        for x in 1:q, z in 1:q
            mul!(Q[x,z,:,:], Qold[x,z,:,:], A[x,x,z,:,:]')
            mul!(Q[x,z,:,:], B[x,x,z,:,:], Qold[x,z,:,:], 1.0,1.0)
            for zz in 1:q
                zz==z && continue
                Q[x,z,:,:] .+= Qold[x,zz,:,:]
            end
            for y in 1:q
                y==x && continue
                mul!(X, B[x,y,z,:,:], Qold[y,z,:,:])
                mul!(Q[x,z,:,:], X, A[x,y,z,:,:]', 1.0, 1.0)
            end
        end
        Q ./= sum(abs, Q)
        if sum(abs2, Q - Qold) < tol_pow
            # @show it
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

    simQ = similar(Q[1,1,:,:])
    simQ1 = similar(Q1[1,1,:,:])
    X = similar(Q1[1,1,:,:])

    for i in eachindex(ηs)
        η = ηs[i]
        maxiter = maxiters[i]
        tol = tols[i]
        for it in 1:maxiter
            findeigen_l!(P, A, B; maxiter_pow, tol_pow, X=simQ)
            findeigen_r!(Q, A, B; maxiter_pow, tol_pow, X=simQ)
            findeigen_l!(P1, A, A; maxiter_pow, tol_pow, X=simQ1)
            findeigen_r!(Q1, A, A; maxiter_pow, tol_pow, X=simQ1)

            n = sum(tr(P[x,z,:,:]'Q[x,z,:,:]) for x in 1:q, z in 1:q)
            P ./= n
            n1 = -2 * sum(tr(P1[x,z,:,:]'Q1[x,z,:,:]) for x in 1:q, z in 1:q)
            P1 ./= n1

            for x in 1:q, y in 1:q, z in 1:q
                # Step proportional to η*parameter
                if x==y
                    mul!(X, P[x,z,:,:]', Q[x,z,:,:])
                    mul!(X, P1[x,z,:,:]', Q1[x,z,:,:], 2.0, 1.0)

                    Anew[x,x,z,:,:] .+= @views η .* sign.(X) .* A[x,x,z,:,:]
                else
                    mul!(simQ1, P1[x,z,:,:]', A[x,y,z,:,:])
                    mul!(X, simQ1, Q1[y,z,:,:])

                    mul!(simQ', P[x,z,:,:]', B[x,y,z,:,:])
                    mul!(X, simQ', Q[y,z,:,:], 1.0, 2.0)

                    Anew[x,y,z,:,:] .+= @views η .* sign.(X) .* A[x,y,z,:,:]
                end

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