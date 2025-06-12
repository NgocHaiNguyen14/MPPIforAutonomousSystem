function sdot = vtol(s,u)
m = 10;
g = 9.81;
fg = [0;0;m*g];
rho = 1.1455;
S = 10;
chord = 5;

Jx = 10;
Jy = 10;
Jz = 12;
Jxz = 5;

J = [Jx, 0 , -Jxz; 0, Jy, 0; -Jxz, 0, Jz];

x = s(1);
y = s(2);
z = s(3);

phi = s(4);
theta = s(5);
psi = s(6);

u = s(7);
v = s(8);
w = s(9);

p = s(10);
q = s(11);
r = s(12);

tl = u(1);
tr = u(2);
al = u(3);
ar = u(4);

Rwb = [cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);...
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi); ...
    -sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)];

posdot = Rwb*[u;v;w];

xdot = posdot(1);
ydot = posdot(2);
zdot = posdot(3);

T = [1, sin(phi)*tan(theta), cos(phi)*tan(theta); ...
    0, cos(phi), -sin(phi); ...
    0, sin(phi)*sec(theta), cos(phi)*sec(theta)];

ang_dot = T*[p;q;r];
phidot = ang_dot(1);
thetadot = ang_dot(2);
psidot = ang_dot(3);

Va = sqrt(u^2 + v^2 + w^2);
fgb = Rwb'*fg;
aeroDymScale =  1/2*rho*Va^2*S;

fdl = aeroDymScale*cdrag(al);
fll = aeroDymScale*clift(al);
fdr = aeroDymScale*cdrag(ar);
flr = aeroDymScale*clift(ar);

fx = tl+tr-fdl-fdr;
fy = 0;
fz = fll+flr;

udot = r*v-q*w + fx/m;
vdot = p*w-r*u + fy/m;
wdot = q*u-p*v + fz/m;

Mx = (fll-flr)*chord/2;
My = aeroDymScale * chord * (cmom(al) + cmom(ar));
Mz = (tr-fdr-tl+fdl)*chord/2;

ang_ddot = inv(J)*([0, r, -q; -r, 0, p; q, -p, 0]*J*[p;q;r] + [Mx;My;Mz]);
pdot = ang_ddot(1);
qdot = ang_ddot(2);
rdot = ang_ddot(3);

sdot = [xdot;ydot;zdot;phidot;thetadot;psidot;udot;vdot;wdot;pdot;qdot;rdot];

end

function coeff = cdrag(angle)

end

function coeff = clift(angle)

end

function coeff = cmom(angle)