function sdot = vtol4_iLQR(s,input)
% VTOL Dynamics with quaternion states

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
q = s(7:10);
omega = s(11:13);

q = q / norm(q + 1e-8); % ensure valid quaternion

% Extract inputs (clamped)
Tr = max(input(1), 0);
Tl = max(input(2), 0);
dr = input(3);
dl = input(4);

% Rotation & aerodynamic forces
R = quat2rotmat(q);
q_dyn = 0.5*rho*norm(vbody)^2;

Fg = R*[0; 0; m*g];
Ft = [Tl+Tr; 0; 0];

[v, alpha, beta] = speedAngles(Ft(1), vbody, vwind, rho, radius, Aprop);

Faero = aeroForces(alpha, beta, v, dr, dl) * q_dyn * Aw;
F = Faero + Fg + Ft;

Maero = aeroMoments(alpha, beta, v, dr, dl, b, c) * q_dyn * Aw;
Mt = [0; 0; (Tr-Tl)*propDist];
M = Maero + Mt;

% Derivatives
pdot = R*vbody;
vbodydot = -cross(omega,vbody) + F/m;
qdot = quaternionDerivative(q, omega);
omegadot = J \ (cross(-omega,J*omega) + M);

sdot = [pdot; vbodydot; qdot; omegadot];
end

function R = quat2rotmat(q)
w = q(1); x = q(2); y = q(3); z = q(4);
R = [1-2*(y^2+z^2), 2*(x*y-w*z), 2*(x*z+w*y);
     2*(x*y+w*z), 1-2*(x^2+z^2), 2*(y*z-w*x);
     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x^2+y^2)];
end

function qdot = quaternionDerivative(q, omega)
w = q(1); x = q(2); y = q(3); z = q(4);
p = omega(1); q_ang = omega(2); r = omega(3);
Q = [-x, -y, -z;
      w, -z,  q_ang;
      z,  w, -p;
     -q_ang,  p,  w];
qdot = 0.5 * Q * omega;
end

function [v, alpha, beta] = speedAngles(T, vbody, vwind, rho, R, Aprop)
vinf = 0;
lp = 0.2;

% ensure positive argument
arg = max(2*T/(rho*Aprop) + vinf^2, 0);
u0 = sqrt(arg)*(1 + (lp/R)/sqrt(1+(lp/R)^2));
u0 = real(u0);

vprop = [0; 0; u0];
v = vprop + vwind + vbody;
v = real(v);

speed_norm = norm(v);
if speed_norm < 1e-8
    alpha = 0; beta = 0;
else
    alpha = atan2(-v(1), v(3));
    beta = asin(max(min(v(2)/speed_norm,1),-1));
end
end

function forces = aeroForces(alpha, beta, v, dr, dl)
L = coeffL(dl) + coeffL(dr);
D = coeffD(dl) + coeffD(dr);
forces = [-D*cos(alpha)+L*sin(alpha); 0; -D*sin(alpha)-L*cos(alpha)];
end

function moments = aeroMoments(alpha, beta, v, dr, dl, b, c)
L = -coeffL(dl) + coeffL(dr);
M = 0; N = 0;
moments = [L*b; M*c; N*b];
end

function coeff = coeffL(delta)
coeff = -0.0004*delta^3 - 0.0006*delta^2 - 0.0007*delta + 0.1629;
end

function coeff = coeffD(delta)
coeff = 0.0008*delta^2 - 0.0007*delta + 0.0059;
end
