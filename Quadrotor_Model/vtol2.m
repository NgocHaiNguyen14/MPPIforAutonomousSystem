function sdot = vtol2(s,input)

% Environment
g = 9.81;
rho = 1.2551;
vwind = [0;0;0];

% VTOL specs
m = 2.23;
Ix = 0.16017;
Iy = 0.04085;
Iz = 0.19866;
Ixz = 0.00008;
propDist = 0.2;
b = 1.2;
c = 0.438;
Aw = 0.478;
radius = 0.2;
Aprop = 0.1; 

J = [Ix, 0, -Ixz; 0, Iy, 0; -Ixz, 0, Iz];

% Extract states
p = s(1:3);
vbody = s(4:6);
E = s(7:9);
omega = s(10:12);

% Extract inputs
Tr = input(1);
Tl = input(2);
dr = input(3);
dl = input(4);

% Calculate forces and moments
R = rotationMatrix(E);
C = euler2Body(E);
q = 1/2*rho*norm(vbody)^2;

Fg = R*[0; 0; m*g];
Ft = [Tl+Tr; 0; 0];

[v, alpha, beta] = speedAngles(Ft(1), vbody, vwind, rho, radius, Aprop);

Faero = aeroForces(alpha, beta, v, dr, dl) * q * Aw;
F = Faero + Fg + Ft;

Maero = aeroMoments(alpha, beta, v, dr, dl, b, c) * q * Aw;
Mt = [0; 0; (Tr-Tl)*propDist];
M = Maero + Mt;

% Take the derivatives
pdot = R*vbody;
vbodydot = -cross(omega,vbody) + F/m;
Edot = inv(C)*omega;
omegadot = inv(J)*(cross(-omega,J*omega) + M);

sdot = [pdot;vbodydot;Edot;omegadot];

end

function R = rotationMatrix(E)
phi = E(1);
theta = E(2);
psi = E(3);

R = [cos(psi)*cos(theta), cos(psi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(psi)*sin(theta)*cos(phi)+sin(psi)*sin(phi);
sin(psi)*cos(theta), sin(psi)*sin(theta)*sin(phi)+cos(psi)*cos(phi), sin(psi)*sin(theta)*cos(phi)-cos(psi)*sin(phi);
-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)];

end

function C = euler2Body(E)
phi = E(1);
theta = E(2);
psi = E(3);

C = [1, 0, -sin(theta); 0, cos(phi), cos(theta)*sin(phi); 0, -sin(phi), cos(theta)*cos(phi)];

end

function [v, alpha, beta] = speedAngles(T, vbody, vwind, rho, R, Aprop)
vinf = 0;
lp = 0.2;

u0 = sqrt(2*T/(rho*Aprop)+vinf^2)*(1+(lp/R)/sqrt(1+(lp/R)^2));
vprop = [0; 0; u0];

v = vprop + vwind + vbody;

speed_norm = norm(v);

if speed_norm < 1e-8
    alpha = 0; % or NaN if you prefer
    beta = 0;
else
    alpha = atan2(-v(1), v(3));
    beta = asin(v(2)/speed_norm);
end

end

function forces = aeroForces(alpha, beta, v, dr, dl)
L = coeffL(dl) + coeffL(dr);

D = coeffD(dl) + coeffD(dr);

Fay = 0;

forces = [-D*cos(alpha)+L*sin(alpha); 0; -D*sin(alpha)-L*cos(alpha)];

end

function moments = aeroMoments(alpha, beta, v, dr, dl, b, c)
L = -coeffL(dl) + coeffL(dr);

M = 0;

N = 0;

moments = [L*b; M*c; N*b];

end

function coeff = coeffL(delta)
coeff = -0.0004*delta^3 + -0.0006*delta^2 + -0.0007*delta + 0.1629;

end

function coeff = coeffD(delta)
coeff = 0.0008*delta^2 + -0.0007*delta + 0.0059;
end

