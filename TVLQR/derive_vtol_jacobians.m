function derive_vtol_jacobians()
% DERIVE_VTOL_JACOBIANS  Symbolically derive df/ds and df/du for the VTOL
% quaternion dynamics and write them to A_jac.m / B_jac.m.
%
% Mirrors vtol_quaternion.m, with two small differences required to make
% the dynamics symbolically differentiable everywhere:
%   * No  q = q/norm(q)  renormalization (assume unit quaternion at trim).
%   * No  if speed_norm < 1e-8  guard (prop-induced flow keeps v_air > 0
%     at every realistic trim point).
%
% Requires the Symbolic Math Toolbox.
%
% Produces, in the current folder:
%   A_jac(s, u)   13x13   df/ds
%   B_jac(s, u)   13x 4   df/du
%
% Example usage AFTER running this once (the generated files persist on
% disk; re-derivation is only needed if you change the model):
%   s0 = zeros(13,1); s0(7) = 1;          % level attitude, zero rates
%   Thover = 2.23*9.81/2;
%   u0 = [Thover; Thover; 0; 0];
%   A = A_jac(s0, u0);
%   B = B_jac(s0, u0);

%% --- Symbolic state and input ---------------------------------------
syms p_x p_y p_z real
syms u_b v_b w_b real
syms q_w q_x q_y q_z real
syms p_r q_r r_r real        % angular rates (subscript to avoid clash with q)
syms Tr Tl dr dl real

params;

s = [p_x; p_y; p_z; u_b; v_b; w_b; q_w; q_x; q_y; q_z; p_r; q_r; r_r];
u = [Tr; Tl; dr; dl];

vbody = [u_b; v_b; w_b];
q     = [q_w; q_x; q_y; q_z];        %#ok<NASGU>
omega = [p_r; q_r; r_r];

%% --- Constants (must match vtol_quaternion.m) ------------------------

J = [Ix, 0, -Ixz; 0, Iy, 0; -Ixz, 0, Iz];

%% --- Rotation matrix (body -> world) --------------------------------
R = [1-2*(q_y^2+q_z^2),   2*(q_x*q_y - q_w*q_z),   2*(q_x*q_z + q_w*q_y);
     2*(q_x*q_y + q_w*q_z), 1-2*(q_x^2+q_z^2),     2*(q_y*q_z - q_w*q_x);
     2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x),   1-2*(q_x^2+q_y^2)];

%% --- Thrust ---------------------------------------------------------
Ft = [Tl + Tr; 0; 0];

%% --- Local airspeed at the wing (prop wash + wind) ------------------
u0_ind = sqrt(2*Ft(1)/(rho*Aprop) + vinf^2) * ...
         (1 + (lp/radius)/sqrt(1 + (lp/radius)^2));
vprop_contrib = [u0_ind; 0; 0];

vwind_body = R.' * vwind;
v_air      = vbody - vwind_body + vprop_contrib;

speed_norm = sqrt(v_air.' * v_air);
alpha      = atan2(v_air(3), v_air(1));
beta       = asin(v_air(2)/speed_norm);   %#ok<NASGU>  (kept for parity)

q_dyn = 0.5 * rho * speed_norm^2;

%% --- Aerodynamic coefficients ---------------------------------------
CL_l = -0.0004*dl^3 - 0.0006*dl^2 - 0.0007*dl + 0.1629;
CL_r = -0.0004*dr^3 - 0.0006*dr^2 - 0.0007*dr + 0.1629;
CD_l =  0.0008*dl^2 - 0.0007*dl + 0.0059;
CD_r =  0.0008*dr^2 - 0.0007*dr + 0.0059;

L_force = CL_l + CL_r;
D_force = CD_l + CD_r;
L_roll  = -CL_l + CL_r;

Faero = [-D_force*cos(alpha) + L_force*sin(alpha);
          0;
         -D_force*sin(alpha) - L_force*cos(alpha)] * q_dyn * Aw;

Maero = [L_roll * b_span;
         0;
         0] * q_dyn * Aw;

%% --- Gravity, propulsion moment, sums -------------------------------
Fg = R.' * [0; 0; m_mass*g];
Mt = [0; 0; (Tr - Tl)*propDist];

F = Faero + Fg + Ft;
M = Maero + Mt;

%% --- Kinematics / dynamics ------------------------------------------
pdot     = R * vbody;
vbodydot = -cross(omega, vbody) + F/m_mass;

Q = [-q_x, -q_y, -q_z;
      q_w, -q_z,  q_y;
      q_z,  q_w, -q_x;
     -q_y,  q_x,  q_w];
qdot = 0.5 * Q * omega;

omegadot = J \ (cross(-omega, J*omega) + M);

sdot = [pdot; vbodydot; qdot; omegadot];

%% --- Jacobians ------------------------------------------------------
fprintf('Computing A = df/ds (13x13) ...\n');
A_sym = jacobian(sdot, s);

fprintf('Computing B = df/du (13x4)  ...\n');
B_sym = jacobian(sdot, u);

% Optional: shorten expressions in the generated files.  Can be slow.
%   A_sym = simplify(A_sym);
%   B_sym = simplify(B_sym);

%% --- Export as numeric MATLAB functions -----------------------------
fprintf('Writing jacobians.m ...\n');

matlabFunction( ...
    A_sym, B_sym, ...
    'Vars', {s, u}, ...
    'File', 'vtol_grads', ...
    'Outputs', {'A', 'B'}, ...
    'Optimize', true);

fprintf('Done.\n');
end
