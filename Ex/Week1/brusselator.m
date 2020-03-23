a = 1;
b = 3;
x0 = [2; 2];
options = odeset('Jacobian',@JacBrusselator,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[T,X]=ode15s(@Brusselator, [0; 50], x0, options, a, b);


function xdot = Brusselator(t,x,a,b)

xdot=zeros(2,1);
xdot(1) = a + x(1)*x(1)*x(2) - (b + 1)*x(1);
xdot(2) = b*x(1) - x(1)*x(1)*x(2);
end

function Jac = JacBrusselator(t,x,a,b)

Jac = zeros(2,2);
Jac(1,1) = 2*x(1)*x(2) - (b + 1);
Jac(1,2) = x(1)*x(1);
Jac(2,1) = b - 2*x(1)*x(2);
Jac(2,2) = -x(1)*x(1);
end