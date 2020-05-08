function rhs = vdpProblJac(t, x, mu)

rhs = [0 1;-2*mu*x(1)*x(2)-1 mu*(1-x(1)*x(1))];

end