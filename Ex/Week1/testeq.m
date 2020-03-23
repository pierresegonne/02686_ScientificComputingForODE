x0 = [2; 2];
lbd = x0(2) / x0(1);
options = odeset('Jacobian',@JacTestEq,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[T,X]=ode15s(@TestEq, [0; 50], x0(1), options, lbd);


function xdot = TestEq(t,x,lbd)

xdot=zeros(1,1);
xdot(1) = lbd*x(1);
end

function Jac = JacTestEq(t,x,lbd)

Jac = zeros(1,1);
Jac(1,1) = lbd;
end