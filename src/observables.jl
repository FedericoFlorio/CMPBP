E(x,y; q=2) = [i==x && j==y for i in 1:q, j in 1:q]
potts2spin(x, i=0; q=2) = (x-1)/(q-1)*2 - 1

function propagator(A,t)
    q = size(A,1)
    d = size(A,4)
    AA = sum(kron(E(x,y;q), E(z,z;q), A[x,y,z,:,:]) for x in 1:q, y in 1:q, z in 1:q)+
        sum(kron(E(x,x;q), E(z,z1;q), I(d)) for x in 1:q, z in 1:q, z1 in 1:q if z1≠z)
    if t < Inf
        exp(t*AA)
    else
        L,V = eigen(AA)
        i = searchsortedfirst(real.(L),real(L[end])-1e-20)
        u = inv(V)[i:end,:]
        v = V[:,i:end]
        v*u
    end
end

function marginals(A, Δts...)
    d, q = size(A,4), size(A,1)
    
    E(z) = kron([x == y == z for x in 1:q, y in 1:q], I(q), I(d))
    Ps = [propagator(A, Δt) for Δt in (Δts..., Inf)]
    P = map(Iterators.product(fill(1:q, length(Ps))...)) do (ys...,)
        prod(E(y)*p for (y,p) in zip(ys, Ps)) |> tr
    end
    P ./ sum(P)
end

function magnetization(A)
    p = marginals(A)
    return sum(potts2spin(x)*p[x] for x in eachindex(p))
end

function logdot(A,B; maxiter=100)
    q = size(A, 1)
    da, db = size(A,4), size(B,4)
    size(A,2)==size(A,3)==size(B,1)==size(B,2)==size(B,3)==q || error("Incompatible dimensions")
    (size(A,5)==da && size(B,5)==db) || error("Incompatible dimensions")

    S = build_S(A,B; q,da,db)
    v = rand(q*q*da*db)
    vold = similar(v)

    for it in 1:maxiter
        vold = copy(v)
        v = S*v
        v ./= norm(v)
        if norm(v - vold) < 1e-16
            break
        end
    end
    return sum(S*v) / sum(v)
end

fidelity(A,B; kw...) = logdot(A,B; kw...) - 0.5 * (logdot(A,A; kw...) + logdot(B,B; kw...))