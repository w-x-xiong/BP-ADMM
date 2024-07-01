function [x, fail] = ADMM(Tx, Rx, Rg, rho, maxiter, delta)

[H, M] = size(Tx);
[~, L] = size(Rx);

x = zeros(H,1);

%b = 0;

d = zeros((M+L),1);

psi = zeros(H*(M+L),1);

nu = zeros(H*(M+L),1);

fail = false;

%k: counter for iterations
k = 0;

while 1
    
    x_old = x;
    
    %x-update
    alpha = zeros(H,M);
    theta = zeros(H,L);

    for m = 1:M
        alpha(:,m) = Tx(:,m) + psi((m-1)*H+1:(m-1)*H+H)*d(m) - (1/rho)*nu((m-1)*H+1:(m-1)*H+H);
    end

    for l = 1:L
        theta(:,l) = Rx(:,l) + psi(H*M+(l-1)*H+1:H*M+(l-1)*H+H)*d(M+l) - (1/rho)*nu(H*M+(l-1)*H+1:H*M+(l-1)*H+H);
    end

    x = (sum(alpha,2) + sum(theta,2))/(M+L);

    %b-update
    z = [];
    for m = 1:M
        for l = 1:L
            z = [z;Rg(m,l) - d(m) - d(M+l)];
        end
    end

    b = lsqnonneg(ones(M*L,1),z);

    %psi-update
    v = zeros(H,M);
    w = zeros(H,L);

    for m = 1:M
        v(:,m) = x - Tx(:,m) + (1/rho)*nu((m-1)*H+1:(m-1)*H+H);
        psi((m-1)*H+1:(m-1)*H+H) = v(:,m)/norm(v(:,m));
    end

    for l = 1:L
        w(:,l) = x - Rx(:,l) + (1/rho)*nu(H*M+(l-1)*H+1:H*M+(l-1)*H+H);
        psi(H*M+(l-1)*H+1:H*M+(l-1)*H+H) = w(:,l)/norm(w(:,l));
    end

    %d-update
    y1 = [];
    y2 = [];
    y3 = [];

    for m = 1:M
        y2 = [y2;sqrt(rho/2)*norm(v(:,m))];
        for l = 1:L
            y1 = [y1;Rg(m,l) - b];
        end
    end

    for l = 1:L
        y3 = [y3;sqrt(rho/2)*norm(w(:,l))];
    end

    y = [y1',y2',y3']';

    A1 = [];

    for m = 1:M
        A1 = [A1;[zeros(L,(m-1)),ones(L,1),zeros(L,M-m),eye(L)]];
    end

    A2 = sqrt(rho/2)*[eye(M),zeros(M,L)];

    A3 = sqrt(rho/2)*[zeros(L,M),eye(L)];

    A = [A1',A2',A3']';

    d = (pinv(A'*A))*A'*y;

    %lambda-update
    for m = 1:M
        nu((m-1)*H+1:(m-1)*H+H) = rho*(x - Tx(:,m) - psi((m-1)*H+1:(m-1)*H+H)*d(m)) + nu((m-1)*H+1:(m-1)*H+H);
    end

    %mu-update
    for l = 1:L
        nu(H*M+(l-1)*H+1:H*M+(l-1)*H+H) = rho*(x - Rx(:,l) - psi(H*M+(l-1)*H+1:H*M+(l-1)*H+H)*d(M+l)) + nu(H*M+(l-1)*H+1:H*M+(l-1)*H+H);
    end
    
    if ((norm(x - x_old)/min(norm(x),norm(x_old))) < delta) || ((k+1) == maxiter)
        if ((k+1) == maxiter)
            fprintf('have reached the maximum iteration number\n')
            fail = true;
        end
        break
    end
    
    k = k + 1;
    
end

fprintf('It takes %d iterations to meet a termination condition\n', k)

end

