function findeigen_l!(P,A,B; maxiter_pow=100, tol_pow=1e-6)
    Pold = copy(P)
    q = size(P,1)
    da, db = size(A,4), size(B,4)

    for it in 1:maxiter_pow
        Pold .= P
        for x in 1:q, z in 1:q
            P[x,z,:,:] = Pold[x,z,:,:]*A[x,x,z,:,:] + B[x,x,z,:,:]'* Pold[x,z,:,:] +
                        sum(Pold[x,zz,:,:] for zz in 1:q if zz≠z) +
                        sum(B[x,y,z,:,:]'*Pold[y,z,:,:]*A[x,y,z,:,:] for y in 1:q if y≠x)
        end
        P ./= sum(abs, P)
        sum(abs2, P - Pold) < tol_pow && return P
    end
    return P
end
function findeigen_r!(Q,A,B; maxiter_pow=100, tol_pow=1e-6)
    Qold = copy(Q)
    q = size(Q,1)
    da, db = size(A,4), size(B,4)

    for it in 1:maxiter_pow
        Qold .= Q
        for x in 1:q, z in 1:q
            Q[x,z,:,:] = Qold[x,z,:,:]*A[x,x,z,:,:]' + B[x,x,z,:,:]*Qold[x,z,:,:] +
                        sum(Qold[x,zz,:,:] for zz in 1:q if zz≠z) +
                        sum(B[x,y,z,:,:]*Qold[y,z,:,:]*A[x,y,z,:,:]' for y in 1:q if y≠x)
        end
        Q ./= sum(abs, Q)
        sum(abs2, Q - Qold) < tol_pow && return Q
    end
    return Q
end

function ascent!(A,B; maxiter=100, η=0.1, tol=1e-12, maxiter_pow=1000, tol_pow=1e-12)
    q = size(A, 1)
    da, db = size(A,4), size(B,4)
    Anew = copy(A)

    Q = rand(q,q,db,da)
    P = rand(q,q,db,da)
    Q1 = rand(q,q,da,da)
    P1 = rand(q,q,da,da)

    for it in 1:maxiter
        findeigen_l!(P, A, B; maxiter_pow, tol_pow)
        findeigen_r!(Q, A, B; maxiter_pow, tol_pow)
        findeigen_l!(P1, A, A; maxiter_pow, tol_pow)
        findeigen_r!(Q1, A, A; maxiter_pow, tol_pow)

        n = sum(tr(P[x,z,:,:]'Q[x,z,:,:]) for x in 1:q, z in 1:q)
        P ./= n
        n1 = -2 * sum(tr(P1[x,z,:,:]'Q1[x,z,:,:]) for x in 1:q, z in 1:q)
        P1 ./= n1

        for x in 1:q, y in 1:q, z in 1:q
            if x==y
                Anew[x,x,z,:,:] .+= @views η .* (P[x,z,:,:]'Q[x,z,:,:] + 2 * P1[x,z,:,:]'Q1[x,z,:,:])
            else
                Anew[x,y,z,:,:] .+= @views η .* (P[x,z,:,:]'B[x,y,z,:,:]*Q[y,z,:,:] + 2 * P1[x,z,:,:]'A[x,y,z,:,:]*Q1[y,z,:,:])
            end
        end
        A .= Anew
    end
end

function ascent(A,B; maxiter=100, η=0.1, tol=1e-12, maxiter_pow=1000, tol_pow=1e-12)
    X = copy(A)
    ascent!(X,B; maxiter, η, tol, maxiter_pow, tol_pow)
    return X
end