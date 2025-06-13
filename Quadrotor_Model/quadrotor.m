function f = quadrotor(s,u)
m = 0.2;
g = 9.81;
J = diag([7*1e-3, 7*1e-3, 12*1e-3]);
e3 = [0;0;1];

x = s(1);
y = s(2);
z = s(3);

roll = s(4);
pitch = s(5);
yaw = s(6);

vx = s(7);
vy = s(8);
vz = s(9);

p = s(10);
q = s(11);
r = s(12);

thrust = u(1);
tx = u(2);
ty = u(3);
tz = u(4);

xdot = vx;
ydot = vy;
zdot = vz;

R = [cos(yaw)*cos(pitch)-sin(roll)*sin(yaw)*sin(pitch), -cos(roll)*sin(yaw), cos(yaw)*sin(pitch)+cos(pitch)*sin(roll)*sin(yaw);...
    cos(pitch)*sin(yaw)+cos(yaw)*sin(roll)*sin(pitch), cos(roll)*cos(yaw), sin(yaw)*sin(pitch)-cos(yaw)*cos(pitch)*sin(roll);...
    -cos(roll)*sin(pitch), sin(roll), cos(roll)*cos(pitch)];

vdot =  R/m*thrust*e3 -g*e3;
vxdot = vdot(1);
vydot = vdot(2);
vzdot = vdot(3);

T = [cos(pitch), 0, -cos(roll)*sin(pitch);...
    0, 1, sin(roll);...
    sin(pitch), 0, cos(roll)*cos(pitch)];

w = inv(T)*[p;q;r];
rolldot = w(1);
pitchdot = w(2);
yawdot = w(3);

wdot = inv(J)*([tx;ty;tz]-cross([p;q;r], J*[p;q;r]));
pdot = wdot(1);
qdot = wdot(2);
rdot = wdot(3);

f = [xdot;ydot;zdot;rolldot;pitchdot;yawdot;vxdot;vydot;vzdot;pdot;qdot;rdot];

end

