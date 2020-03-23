mu = 1;
x0 = [2.0; 0.0];
tspan= [0; 5*mu];
options = odeset('Jacobian',@JacVanDerPol,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[Tode,Xode]=ode15s(@VanDerPol, tspan, x0, options, mu);
[X,T] = ImplicitEuleurAdaptiveStep(@VanderPolFunJac, tspan, x0, 1, 1.0e-6, 1.0e-6, [mu]);

function [X,T] = ImplicitEuleurAdaptiveStep(fJ, tspan, x0, h0, abstol, reltol, varargin)
    % Error Controlling
    epstol = 0.8;
    facmax = 5;
    facmin = 0.1;
    
    % Newtons Thresholds
    maxit = 100;
    newtownstol = 1e-8;
    
    % Integration Interval
    t0 = tspan(1);
    tf = tspan(2);
    
    % Initial Conditions
    h = h0;
    t = t0;
    x = x0;
    
    % Storage
    T = t;
    X = x';
    
    while t < tf
        if (t+h>tf)
            h = tf - t;
        end
        
        [fval,~] = feval(fJ, t, x, varargin{:});
        
        acceptStep = false;
        while ~acceptStep
            % Take step of size h
            x1init = x + h*fval;
            x1 = NewtonsMethodODE(fJ, t+h, x, h, x1init, newtownstol, maxit, varargin{:});
            
            % Take 2 steps of size h/2
            x_hat12init = x + (h/2)*fval;
            t_hat12 = t + (h/2);
            x_hat12 = NewtonsMethodODE(fJ, t_hat12, x, h/2, x_hat12init, newtownstol, maxit, varargin{:});
            
            [fval_hat12, ~] = feval(fJ, t_hat12, x_hat12, varargin{:});
            x_hat1init = x_hat12 + (h/2)*fval_hat12;
            x_hat1 = NewtonsMethodODE(fJ, t+h, x_hat12, h/2, x_hat1init, newtownstol, maxit, varargin{:});
            
            % Error estimation
            e = abs(x1 - x_hat1);
            r = max(e ./ max(abstol, abs(x_hat1)*reltol));
            
            acceptStep = (r <= 1.0);
            if acceptStep
                t = t + h;
                x = x_hat1;
                
                T = [T;t];
                X = [X;x'];
            end
            
            % Update h
            h = max(facmin, min(sqrt(epstol/r), facmax))*h;
        end
    end
end

function x = NewtonsMethodODE(FunJac, tk, xk, dt, xinit, tol, maxit, varargin)
    k = 0;
    x = xinit;
    t = tk + dt;
    [f,J] = feval(FunJac,t,x,varargin{:});
    R = x - dt*f - xk;
    I = eye(size(xk,1));
    while( (k < maxit) && (norm(R,'inf') > tol) )
        k = k+1;
        dRdx = I - dt*J;
        dx = dRdx\R;
        x = x - dx;
        [f,J] = feval(FunJac,t,x,varargin{:});
        R = x - dt*f - xk;
    end
end



function xdot = VanDerPol(t,x,mu)

xdot=zeros(2,1);
xdot(1) = x(2);
xdot(2) = mu*(1-x(1)*x(1))*x(2) - x(1);
end

function Jac = JacVanDerPol(t,x,mu)

Jac = zeros(2,2);
Jac(1,2) = 1;
Jac(2,1) = -2*mu*x(1)*x(2)-1;
Jac(2,2) = mu*(1-x(1)*x(1));
end

function [f,Jac] = VanderPolFunJac(t,x,mu)
    f = zeros(2,1);
    f(1) = x(2);
    f(2) = mu*(1-x(1)*x(1))*x(2)-x(1);
    Jac = zeros(2,2);
    Jac(1,2) = 1;
    Jac(2,1) = -2*mu*x(1)*x(2)-1;
    Jac(2,2) = mu*(1-x(1)*x(1));
end