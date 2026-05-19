function sdot = vtol_quaternion(s, input)
% VTOL_QUATERNION  6-DOF dynamics of a VTOL using quaternion attitude.
%
% State vector s (13x1):
%   s(1:3)   = position in world frame (NED: x-north, y-east, z-down)
%   s(4:6)   = body-frame linear velocity [u; v; w]
%   s(7:10)  = attitude quaternion [w; x; y; z]   (body wrt. world)
%   s(11:13) = body-frame angular velocity [p; q; r]
%
% Input vector input (4x1):
%   input(1) = Tr  right propeller thrust
%   input(2) = Tl  left  propeller thrust
%   input(3) = dr  right elevon deflection (deg)
%   input(4) = dl  left  elevon deflection (deg)
%
% Body axes convention:
%   x = forward (prop / thrust axis; "up" in hover for a tail-sitter)
%   y = right
%   z = down

% --- Environment -------------------------------------------------------
g     = 9.81;
rho   = 1.2551;
vwind = [0; 0; 0];   % wind expressed in the WORLD frame

% --- VTOL specs --------------------------------------------------------
m        = 2.23;
Ix       = 0.16017;
Iy       = 0.04085;
Iz       = 0.19866;
Ixz      = 0.00008;
propDist = 0.2;
b        = 1.2;
c        = 0.438;
Aw       = 0.478;
radius   = 0.2;
Aprop    = 0.1;

J = [Ix, 0, -Ixz; 0, Iy, 0; -Ixz, 0, Iz];

% --- Extract states ----------------------------------------------------
p_pos = s(1:3);
vbody = s(4:6);
q     = s(7:10);
omega = s(11:13);

q = q / norm(q);   % keep unit quaternion

% --- Extract inputs ----------------------------------------------------
Tr = input(1);
Tl = input(2);
dr = input(3);
dl = input(4);

% --- Rotation matrix (body -> world) -----------------------------------
R = quat2rotmat(q);

% --- Thrust (body frame, along +x) -------------------------------------
Ft = [Tl + Tr; 0; 0];

% --- Local airspeed at the wing (includes prop wash and wind) ----------
% NOTE passes R so that wind, given in WORLD frame, is rotated into body.
[v_air, alpha, beta] = speedAngles(Ft(1), vbody, vwind, R, rho, radius, Aprop);

% --- Dynamic pressure: use LOCAL airspeed at wing, not raw vbody -------
q_dyn = 0.5 * rho * norm(v_air)^2;

% --- Gravity in BODY frame --------------------------------------------
% R is body->world, so its transpose maps world vectors into body.
Fg = R.' * [0; 0; m*g];

% --- Aerodynamic forces and moments -----------------------------------
Faero = aeroForces (alpha, beta, v_air, dr, dl)       * q_dyn * Aw;
Maero = aeroMoments(alpha, beta, v_air, dr, dl, b, c) * q_dyn * Aw;

% Differential thrust -> yaw moment about body z.
% (Assumes right motor at +y, left motor at -y; flip the sign if reversed.)
Mt = [0; 0; (Tr - Tl) * propDist];

F = Faero + Fg + Ft;
M = Maero + Mt;

% --- Kinematics / dynamics --------------------------------------------
pdot     = R * vbody;
vbodydot = -cross(omega, vbody) + F / m;
qdot     = quaternionDerivative(q, omega);
omegadot = J \ (cross(-omega, J*omega) + M);     % prefer backslash to inv()

sdot = [pdot; vbodydot; qdot; omegadot];

end

% =======================================================================
function R = quat2rotmat(q)
% Body -> world rotation matrix.  q = [w; x; y; z].
w = q(1); x = q(2); y = q(3); z = q(4);
R = [1-2*(y^2+z^2),   2*(x*y - w*z),   2*(x*z + w*y);
     2*(x*y + w*z),   1-2*(x^2+z^2),   2*(y*z - w*x);
     2*(x*z - w*y),   2*(y*z + w*x),   1-2*(x^2+y^2)];
end

% =======================================================================
function qdot = quaternionDerivative(q, omega)
% Quaternion kinematics:  qdot = 0.5 * q (x) [0; omega], q = [w;x;y;z].
%
% FIX vs. original: the matrix entries below are QUATERNION components only.
% The previous version accidentally substituted angular-rate variables
% (p, q_ang) where (x, y) belong.
w = q(1); x = q(2); y = q(3); z = q(4);

Q = [-x, -y, -z;
      w, -z,  y;
      z,  w, -x;
     -y,  x,  w];

qdot = 0.5 * Q * omega;
end

% =======================================================================
function [v, alpha, beta] = speedAngles(T, vbody, vwind, R, rho, R_prop, Aprop)
% Airspeed vector seen at the wing, expressed in body frame.
%
%   T       total thrust magnitude
%   vbody   body-frame velocity of aircraft
%   vwind   WORLD-frame wind velocity
%   R       body->world rotation matrix
%   R_prop  propeller radius
%   Aprop   propeller disk area

vinf = 0;
lp   = 0.2;     % distance from disk to wing along prop axis

% Induced axial flow speed through the disk (momentum theory + position factor)
% Clip T to >= 0: momentum theory assumes positive thrust; sqrt() of a
% negative argument would return a complex number and break atan2/asin below.
T_eff = max(T, 0);
u0 = sqrt(2*T_eff/(rho*Aprop) + vinf^2) * (1 + (lp/R_prop)/sqrt(1 + (lp/R_prop)^2));

% Thrust along +body_x  =>  slipstream travels in -body_x  =>  the wing
% sees air flowing past in -body_x  =>  the aircraft-relative-to-air
% velocity vector picks up a +x contribution from the prop.
vprop_contrib = [u0; 0; 0];     % FIX vs. original [0;0;u0]

% Wind: transform from world into body frame.  FIX vs. original (raw add).
vwind_body = R.' * vwind;

% Airspeed = (aircraft velocity) - (air velocity), both in body frame.
v = vbody - vwind_body + vprop_contrib;

speed_norm = norm(v);
if speed_norm < 1e-8
    alpha = 0;
    beta  = 0;
else
    alpha = atan2(v(3), v(1));      % FIX: standard aero AoA = atan2(w, u)
    beta  = asin(v(2)/speed_norm);  % sideslip
end
end

% =======================================================================
function forces = aeroForces(alpha, beta, v, dr, dl) %#ok<INUSD>
% Wind-frame force [-D; 0; -L] rotated by alpha into body axes.
L = coeffL(dl) + coeffL(dr);
D = coeffD(dl) + coeffD(dr);

forces = [-D*cos(alpha) + L*sin(alpha);
           0;
          -D*sin(alpha) - L*cos(alpha)];
end

% =======================================================================
function moments = aeroMoments(alpha, beta, v, dr, dl, b, c) %#ok<INUSD,INUSL>
% Roll moment from differential elevon deflection; pitch/yaw left at 0.
L_roll  = -coeffL(dl) + coeffL(dr);
M_pitch = 0;
N_yaw   = 0;
moments = [L_roll*b; M_pitch*c; N_yaw*b];
end

% =======================================================================
function coeff = coeffL(delta)
coeff = -0.0004*delta^3 - 0.0006*delta^2 - 0.0007*delta + 0.1629;
end

% =======================================================================
function coeff = coeffD(delta)
coeff = 0.0008*delta^2 - 0.0007*delta + 0.0059;
end