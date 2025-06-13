function sdot = quadload(s,u)
params.mq = 0.716;
params.ml = 0.2;
params.Ix = 7e-3;
params.Iy = 7e-3;
params.Iz = 12e-3;
params.Il = 1e-6;
params.l = 1;

q = s(1:8);
qdot = s(9:16);

M = inertiaMatrix(q, params)+eye(8)*1e-3;
C = coriolisMatrix(q, qdot, params);
G = gravityVector(q, params);

b = controlMatrix(q);

qddot = inv(M)*(-C*qdot-G+b*u);

sdot = [qdot;qddot];

end

function M = inertiaMatrix(q, params)
x = q(1);
y = q(2);
z = q(3);

phi = q(4);
theta = q(5);
psi = q(6);

alpha = q(7);
beta = q(8);

mq = params.mq;
ml = params.ml;
l = params.l;
Ix = params.Ix;
Iy = params.Iy;
Iz = params.Iz;
Il = params.Il;

M = zeros(8,8);
M(1,1) = mq + ml;
M(2,2) = mq + ml;
M(3,3) = mq + ml;

M(1,7) = ml*l*cos(alpha)*cos(beta);
M(7,1) = M(1,7);

M(1,8) = -ml*l*sin(alpha)*sin(beta);
M(8,1) = M(1,8);

M(2,7) = ml*l*cos(alpha)*sin(beta);
M(7,2) = M(2,7);

M(2,8) = ml*l*sin(alpha)*cos(beta);
M(8,2) = M(2,8);

M(3,7) = ml*l*sin(alpha);
M(7,3) = M(3,7);

M(4,4) = Iz*sin(theta)^2 + (cos(theta)^2)*(Iy*sin(phi)^2+Iz*cos(phi)^2);

M(4,5) = (Iy-Ix)*cos(theta)*cos(phi)*sin(phi);
M(5,4) = M(4,5);

M(4,6) = -Iz*sin(theta);
M(6,4) = M(4,6);

M(5,5) = Iy*cos(phi)^2 + Iz*sin(phi)^2;

M(6,6) = Iz;

M(7,7) = ml*l^2 + Il;

M(8,8) = ml*l^2*sin(alpha)^2 + Il;


end

function C = coriolisMatrix(q, qdot, params)
x = q(1);
y = q(2);
z = q(3);

phi = q(4);
theta = q(5);
psi = q(6);

alpha = q(7);
beta = q(8);

xdot = qdot(1);
ydot = qdot(2);
zdot = qdot(3);

phidot = qdot(4);
thetadot = qdot(5);
psidot = qdot(6);

alphadot = qdot(7);
betadot = qdot(8);

mq = params.mq;
ml = params.ml;
l = params.l;
Ix = params.Ix;
Iy = params.Iy;
Iz = params.Iz;
Il = params.Il;

C = zeros(8,8);
C(1,7) = -ml*l*(cos(alpha)*sin(beta)*betadot + sin(alpha)*cos(beta)*alphadot);
C(1,8) = -ml*l*(cos(alpha)*sin(beta)*alphadot + sin(alpha)*cos(beta)*betadot);

C(2,7) = ml*l*(cos(alpha)*cos(beta)*betadot - sin(alpha)*sin(beta)*alphadot);
C(2,8) = ml*l*(cos(alpha)*cos(beta)*alphadot - sin(alpha)*sin(beta)*betadot);

C(3,7) = ml*l*cos(alpha)*alphadot;

C(4,4) = Iz*thetadot*sin(theta)*cos(theta) - (Iy+Ix)*thetadot*sin(theta)*cos(theta)*sin(phi)^2 + ...
    (Iy-Ix)*phidot*cos(theta)^2*sin(phi)*cos(phi);
C(4,5) = Iz*psidot*sin(theta)*cos(theta) - (Iy-Ix)*(thetadot*sin(theta)*cos(phi)*sin(phi)+phidot*cos(theta)*sin(phi)^2)- ...
    (Iy+Ix)*(psidot*sin(theta)*cos(theta)*cos(phi)^2-phidot*cos(theta)*cos(phi)^2);

C(4,6) = -Iz*thetadot*cos(theta)+(Iy-Ix)*psidot*cos(theta)^2*cos(phi)*sin(phi);

C(5,4) = psidot*cos(theta)*sin(theta)*(-Iz+Iy*sin(phi)^2+Ix*cos(phi)^2);
C(5,6) = Iz*psidot*cos(theta)+(Iy-Ix)*(-thetadot*sin(theta)*cos(phi)+psidot*cos(theta)*cos(phi)^2-psidot*cos(theta)*sin(phi)^2);

C(6,4) = -(Iy-Ix)*psidot*cos(theta)^2*cos(phi)*sin(phi);
C(6,5) = -Iz*psidot*cos(theta)+(Iy-Ix)*(thetadot*sin(phi)*cos(phi)+ ...
    psidot*cos(theta)*sin(phi)^2-psidot*cos(theta)*sin(phi)^2);

C(7,8) = -ml*l^2*sin(alpha)*cos(alpha)*betadot;
C(8,7) = -C(7,8);
C(8,8) = ml*l*sin(alpha)*cos(alpha)*alphadot;

end

function G = gravityVector(q, params)
mq = params.mq;
ml = params.ml;
l = params.l;

g = 9.81;

alpha = q(7);

G = [0;0;(mq+ml)*g;0;0;0;ml*l*g*sin(alpha);0];

end

function B = controlMatrix(q)
phi = q(4);
theta = q(5);
psi = q(6);

B = zeros(8,4);

B(1,1) = sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta);
B(2,1) = cos(phi)*sin(theta)*sin(psi)-cos(psi)*sin(phi);
B(3,1) = cos(theta)*cos(phi);

B(4,2) = 1;
B(5,3) = 1;
B(6,4) = 1;

end