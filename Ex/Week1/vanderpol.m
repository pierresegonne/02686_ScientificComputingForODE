mu = 10;
x0 = [2.0; 0.0];
options = odeset('Jacobian',@JacVanDerPol,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[T,X]=ode15s(@VanDerPol, [0; 5*mu], x0, options, mu);


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
