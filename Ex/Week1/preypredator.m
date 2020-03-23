a = 1;
b = 1;
x0 = [2; 2];
options = odeset('Jacobian',@JacPreyPredator,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[T,X]=ode15s(@PreyPredator, [0; 50], x0, options, a, b);


function xdot = PreyPredator(t,x,a,b)

xdot=zeros(2,1);
xdot(1) = a*(1-x(2))*x(1);
xdot(2) = -b*(1-x(1))*x(2);
end

function Jac = JacPreyPredator(t,x,a,b)

Jac = zeros(2,2);
Jac(1,1) = a*(1-x(2));
Jac(1,2) = -a*x(1);
Jac(2,1) = b*x(2);
Jac(2,2) = -b*(1-x(1));
end