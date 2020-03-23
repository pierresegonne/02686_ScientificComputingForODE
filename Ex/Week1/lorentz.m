beta = 8/3;
rho = 28;
sigma = 10;
eta = sqrt(rho*(beta-1));
xc1 = zeros(3,1);
xc1(1) = 1/rho;
xc1(2) = eta;
xc1(3) = eta;
xc2 = zeros(3,1);
x0 = xc1 + xc2;
options = odeset('Jacobian',@JacLorentz,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[T,X]=ode15s(@Lorentz, [0; 50], x0, options, beta, rho, sigma);


function xdot = Lorentz(t,x,beta,rho,sigma)

xdot=zeros(3,1);
xdot(1) = sigma*(x(2)-x(1));
xdot(2) = x(1)*(rho-x(3))-x(2);
xdot(3) = x(1)*x(2)-beta*x(3);
end

function Jac = JacLorentz(t,x,beta,rho,sigma)

Jac = zeros(3,3);
Jac(1,1) = -sigma;
Jac(1,2) = 0;
Jac(1,3) = sigma;
Jac(2,1) = rho;
Jac(2,2) = -1;
Jac(2,3) = -x(1);
Jac(3,1) = x(2);
Jac(3,2) = x(1);
Jac(3,3) = -beta;
end