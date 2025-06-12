function sdot = vtol1(s,input)
m = 0.9;
g = -9.81;
Fg = [0;0;m*g];
J = diag([0.023, 0.02, 0.033]);
rho = 1.1455; % Air density
S = 0.22; % Area of wing
c = 0.26; % Chord
b = 0.9; % Wingspan
alpha = 0;
beta = 0; %Wind angle

KT = 2.015*1e-6;
KM = 2.444*1e-10;

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

rot_spd = input(1);
ar = input(2);
al = input(3);

Reb = [cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);...
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi); ...
    -sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)];

pos_dot = Reb*[u;v;w];

xdot = pos_dot(1); %
ydot = pos_dot(2); %
zdot = pos_dot(3); %

T = [1, sin(phi)*tan(theta), cos(phi)*tan(theta); ...
    0, cos(phi), -sin(phi); ...
    0, sin(phi)*sec(theta), cos(phi)*sec(theta)];

ang_dot = T*[p;q;r];
phidot = ang_dot(1); %
thetadot = ang_dot(2); %
psidot = ang_dot(3); %


deflect = [1, 1; -1, 1]*[ar;al]; % Deflection angles of 2 wings based on input AOA

Va = sqrt(u^2 + v^2 + w^2); % Air speed

Faw = 1/2*rho*(Va^2)*S*forceCoeffs(alpha, beta, deflect, p, q, r, Va, c, b); 

Rbw = [cos(alpha)*cos(beta), -cos(alpha)*sin(beta), -sin(alpha); ...
        sin(beta), cos(beta), 0; ...
        cos(beta)*sin(alpha), -sin(alpha)*sin(beta), cos(alpha)];

Fa = Rbw*[-1, 0, 0; 0, 1, 0; 0, 0, 1]*Faw;
Ft = [KT*rot_spd^2;0;0];

Fb = Ft + Reb'*Fg + Fa;

veldot = Fb/m - cross([p;q;r], [u;v;w]);
udot = veldot(1); %
vdot = veldot(2); %
wdot = veldot(3); %

Maw = 1/2*rho*(Va^2)*S*momentCoeffs(alpha, beta, deflect, p, q, r, Va, c, b);
Mt = [KM*rot_spd^2;0;0];

Mb = Mt + Rbw*Maw;

angvel_dot = inv(J)*(-cross([p;q;r],J*[p;q;r]) + Mb);
pdot = angvel_dot(1); %
qdot = angvel_dot(2); %
rdot = angvel_dot(3); %

sdot = [xdot;ydot;zdot;phidot;thetadot;psidot;udot;vdot;wdot;pdot;qdot;rdot];
end

function coeffs = forceCoeffs(alpha, beta, deflect, p, q, r, Va, c, b)
delta_e = deflect(1);
delta_a = deflect(2);

if Va == 0
    coeffs = [0;0;0];
else
    CD = 0.0208 + 0.0084*alpha + 1.3225*alpha^2 + 0.2*delta_e^2 + 0.0796*beta^2 + -0.0001*beta; % missing CDq*c/(2*Va)*q
    CY = -0.1285*beta + -0.0292*b/(2*Va)*p + -0.0355*b/(2*Va)*r + 0.0299*delta_a; % missing CY0
    CL = 0.0389 + 3.2684*alpha + 6.1523*c/(2*Va)*q + 0.7237*delta_e;
    
    coeffs = [CD;CY;CL];
end

end

function coeffs = momentCoeffs(alpha, beta, deflect, p, q, r, Va, c, b)
delta_e = deflect(1);
delta_a = deflect(2);

if Va == 0
    coeffs = [0;0;0];
else
    Cl = -0.0345*beta + -0.3318*b/(2*Va)*p + 0.0304*b/(2*Va)*r + 0.182*delta_a; % missing Cl0
    Cm = -0.0112 + -0.2625*alpha + -1.8522*c/(2*Va)*q + -0.2845*delta_e;
    Cn = 0.0252*beta + -0.0192*b/(2*Va)*r + -0.0102*delta_a; % missing Cn0 + Cnp*b/(2*Va)*p
    
    coeffs = [b*Cl;c*Cm;b*Cn];
end

end