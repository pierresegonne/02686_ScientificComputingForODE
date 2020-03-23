mu = 1;
x0 = [2.0; 0.0];
tspan= [0; 5*mu];
options = odeset('Jacobian',@JacVanDerPol,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[Tode,Xode]=ode15s(@VanDerPol, tspan, x0, options, mu);
[X,T] = ExplicitEulerAdaptiveStep(@VanDerPol, tspan, x0, 1, 1.0e-6, 1.0e-6, [mu]);


function [X,T] = ExplicitEulerAdaptiveStep(f, tspan, x0, h0, abstol, reltol, varargin)
    % Error Controlling
    epstol = 0.8;
    facmax = 5;
    facmin = 0.1;
    
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
        
        fval = feval(f, t, x, varargin{:});
        
        acceptStep = false;
        while ~acceptStep
            % Take step of size h
            x1 = x + h*fval;
            
            % Take 2 steps of size h/2
            x_hat12 = x + (h/2)*fval;
            t_hat12 = t + (h/2);
            fval_hat12 = feval(f, t_hat12, x_hat12, varargin{:});
            x_hat1 = x_hat12 + (h/2)*fval_hat12;
            
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