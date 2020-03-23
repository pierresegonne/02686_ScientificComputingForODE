Ns = 10;
N = 1000;
T = 10;

[W, Tw, dW] = ScalarStdWienerProcess(T, N, Ns, 100);
X = zeros(size(W));
for s=1:Ns
    X(s,:,:) = SDEEulerExplicitExplicit(@GeometricBrownianMotionF, @GeometricBrownianMotionG, Tw, 1, W(s,:), [0.15,0.15]);
end

function X = SDEEulerExplicitExplicit(f, g, T, x0, W, varargin)
    N = size(T,2) - 1;
    nX = size(x0,1);
    X = zeros(nX, N+1);
    
    X(:,1) = x0;
    for k=1:N
        dt = T(k+1)-T(k);
        dW = W(:,k+1) - W(:,k);
        fval = feval(f, T(k), X(:,k), varargin{:});
        gval = feval(g, T(k), X(:,k), varargin{:});
        X(:,k+1) = X(:,k) + fval*dt + gval*dW;
    end
    
    X = X';
end

function [W, Tw, dW] = ScalarStdWienerProcess(T, N, Ns, seed)
    if nargin == 4
        rng(seed);
    end
    dt = T/N;
    dW = sqrt(dt)*randn(Ns,N);
    W = [zeros(Ns,1) cumsum(dW,2)];
    Tw = 0:dt:T;
end

function f = GeometricBrownianMotionF(t, x, params)
    f = params(1)*x;
end

function g = GeometricBrownianMotionG(t, x, params)
    g = params(2)*x;
end