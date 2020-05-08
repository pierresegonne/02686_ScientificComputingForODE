function rhs = vdpProbl(t, x, mu)



rhs(1) = x(2);
rhs(2) = mu * (1 -x(1)^2)*x(2) - x(1);