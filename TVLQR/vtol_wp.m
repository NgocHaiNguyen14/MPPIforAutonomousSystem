clear; clc;

DYNAMICS = @vtol_quaternion;

nX = 13; % number of states (quaternion representation)
nU = 4;  % number of inputs

% --- Waypoints ---
% [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
waypoints = {
    [0;0;0;    0;0;0; 0.7071;0;0.7071;0; 0;0;0],  % x0 start
    [0;0;5;    0;0;0; 0.7071;0;0.7071;0; 0;0;0],  % x1
    [0;0;5;    0;0;0; 1;0;0;0;           0;0;0],  % x2
    [3;3;6;    0;0;0; 1;0;0;0;           0;0;0],  % x3
    [6;6;8;    0;0;0; 1;0;0;0;           0;0;0],  % x4
    [9;9;9;    0;0;0; 1;0;0;0;           0;0;0],  % x5
    [9;9;9;    0;0;0; 0.7071;0;0.7071;0; 0;0;0],  % x6
    [9;9;0;    0;0;0; 0.7071;0;0.7071;0; 0;0;0],  % xd
};
num_waypoints = length(waypoints);

% --- Per-segment N and dt ---
seg_params = {
    [150, 0.01],  % x0 -> x1
    [50, 0.01],   % x1 -> x2
    [250, 0.01],  % x2 -> x3
    [250, 0.01],  % x3 -> x4
    [250, 0.01],  % x4 -> x5
    [50, 0.01],   % x5 -> x6
    [150, 0.01],  % x6 -> xd
};

% --- CEM Parameters ---
num_samples = 1000;
num_elites  = 50;
iterations  = 500;

% --- CEM smoothing (Option 1: momentum on mean and variance) ---
% alpha = 1 reproduces the original "replace" behavior.
% Lower alpha = more momentum = slower collapse of sigma2.
alpha_mean  = 0.5;   % typical range: 0.3 - 0.7
alpha_sigma = 0.2;   % typical range: 0.2 - 0.5

% =======================================================================
% --- PER-SEGMENT COST MATRICES ---
% Each cell holds a struct with fields: Q (running state), Qf (terminal
% state), R (control). Build them from weight vectors with buildDiagCost().
% State error vector ordering is [pos(3); vel(3); q_err(4); omega(3)].
% =======================================================================
seg_costs = cell(1, num_waypoints - 1);

% --- Segment 1: x0 -> x1  (vertical takeoff, climb to z=5) ---
% Priorities: altitude (z) tracking, hold attitude (still tilted), no spin.
seg_costs{1} = struct( ...
    'Q',  buildDiagCost([2.5, 2.5, 20], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1]), ...
    'Qf', buildDiagCost([100, 100, 100],     [1, 1, 1],    [1000, 1000, 1000, 1000],     [1, 1, 1]), ...
    'R',  0 * eye(nU));

% --- Segment 2: x1 -> x2  (rotate to upright, hover in place) ---
% Priorities: attitude (quaternion) is the maneuver; pin position hard.
seg_costs{2} = struct( ...
    'Q',  buildDiagCost([1.0, 1.0, 1.0],   [0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5]), ...
    'Qf', buildDiagCost([1, 1, 1],     [1, 1, 1],    [100, 100, 100, 100], [10, 10, 10]), ...
    'R',  0 * eye(nU));

% --- Segment 3: x2 -> x3  (translate diagonally upward, upright cruise) ---
% Priorities: position tracking dominates; keep upright; cheap control.
seg_costs{3} = struct( ...
    'Q',  buildDiagCost([2.5, 2.5, 2.5], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1]), ...
    'Qf', buildDiagCost([1000, 1000, 1000],      [10, 10, 10],    [50, 50, 50, 50],     [10, 10, 10]), ...
    'R',  0 * eye(nU));

% --- Segment 4: x3 -> x4  (continued cruise) ---
seg_costs{4} = struct( ...
    'Q',  buildDiagCost([2.5, 2.5, 2.5], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1]), ...
    'Qf', buildDiagCost([1000, 1000, 1000],      [10, 10, 10],    [50, 50, 50, 50],     [10, 10, 10]), ...
    'R',  0 * eye(nU));

% --- Segment 5: x4 -> x5  (final cruise to top of profile) ---
seg_costs{5} = struct( ...
    'Q',  buildDiagCost([2.5, 2.5, 2.5], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1]), ...
    'Qf', buildDiagCost([1000, 1000, 1000],      [10, 10, 10],    [50, 50, 50, 50],     [10, 10, 10]), ...
    'R',  0 * eye(nU));

% --- Segment 6: x5 -> x6  (rotate to landing attitude, hover) ---
% Mirror of segment 2: attitude maneuver, pin position.
seg_costs{6} = struct( ...
    'Q',  buildDiagCost([1.0, 1.0, 1.0],   [0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5]), ...
    'Qf', buildDiagCost([1, 1, 1],     [1, 1, 1],    [100, 100, 100, 100], [10, 10, 10]), ...
    'R',  0 * eye(nU));

% --- Segment 7: x6 -> xd  (vertical landing to z=0) ---
% Priorities: z tracking, hold attitude, kill final velocity (heavy Qf vel).
seg_costs{7} = struct( ...
    'Q',  buildDiagCost([2.5, 2.5, 20], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1]), ...
    'Qf', buildDiagCost([1000, 1000, 1000],     [10, 10, 10],    [1000, 1000, 1000, 1000],     [10, 10, 10]), ...
    'R',  0.1 * eye(nU));

% --- Storage for full trajectory ---
x_full = waypoints{1};   % [13 x T] — grows as segments complete
u_full = [];             % [4  x T] — grows as segments complete

% Track which time indices belong to which segment (for coloring plot)
seg_breaks = [1];        % column index in x_full where each segment starts

% --- Loop over waypoint segments ---
for seg = 1:(num_waypoints - 1)

    N  = seg_params{seg}(1);
    dt = seg_params{seg}(2);

    x0_seg = waypoints{seg};
    xd_seg = waypoints{seg + 1};

    % --- Pull this segment's cost matrices ---
    Q      = seg_costs{seg}.Q;
    Qf     = seg_costs{seg}.Qf;
    R_ctrl = seg_costs{seg}.R;

    fprintf('\n=== Segment %d -> %d  (N=%d, dt=%.4f) ===\n', seg, seg+1, N, dt);

    U      = zeros(nU, N);
    sigma2 = 20 * ones(1, N);

    sigma2_min_history = zeros(1, iterations);
    J_history          = zeros(1, iterations);

    for i = 1:iterations

        J   = zeros(num_samples, 1);
        udu = zeros(nU, N, num_samples);

        for j = 1:num_samples
            du         = sqrt(sigma2) .* randn(nU, N);
            udu(:,:,j) = U + du;

            [J(j), ~, ~] = sampleTrajectoryCosts(x0_seg, xd_seg, udu(:,:,j), ...
                                                  Q, Qf, R_ctrl, dt, N, DYNAMICS);
        end

        [~, Ie]   = mink(J, num_elites);
        [~, Imin] = min(J);

        uOpt = udu(:,:,Imin);
        udue = udu(:,:,Ie);

        Unew      = zeros(nU, N);
        sigma2new = zeros(1, N);
        for j = 1:N
            elite_slice  = squeeze(udue(:, j, :));
            Unew(:, j)   = mean(elite_slice, 2);
            diffs        = elite_slice - Unew(:, j);
            sigma2new(j) = mean(diffs(:).^2);
        end

        % -----------------------------------------------------------------
        % --- Option 1: smoothing / momentum on mean and variance (ACTIVE)
        % -----------------------------------------------------------------
        Unew      = alpha_mean  * Unew      + (1 - alpha_mean)  * U;
        sigma2new = alpha_sigma * sigma2new + (1 - alpha_sigma) * sigma2;

        % -----------------------------------------------------------------
        % --- Option 2: scheduled (decaying) variance floor (COMMENTED)
        % -----------------------------------------------------------------
        % Replace the constant sigma2_min below with one of these schedules.
        % Keep the floor high early (forces exploration) and let it relax late.
        %
        % % Exponential decay (smooth):
        % sigma2_floor_start = 5.0;
        % sigma2_floor_end   = 0.05;
        % tau                = 100;
        % sigma2_min = sigma2_floor_end + ...
        %              (sigma2_floor_start - sigma2_floor_end) * exp(-i / tau);
        %
        % % Linear decay:
        % sigma2_floor_start = 5.0;
        % sigma2_floor_end   = 0.05;
        % sigma2_min = max(sigma2_floor_end, ...
        %     sigma2_floor_start - (sigma2_floor_start - sigma2_floor_end) * i / iterations);
        %
        % % Power-law (slower tail):
        % sigma2_floor_start = 5.0;
        % tau = 50;
        % sigma2_min = sigma2_floor_start / (1 + i / tau);

        % -----------------------------------------------------------------
        % --- Option 3: additive decaying noise injection (COMMENTED)
        % -----------------------------------------------------------------
        % noise_inject = max(1e-3, 2.0 * 0.99^i);
        % sigma2new    = sigma2new + noise_inject;

        sigma2_min = 1e-1;     % constant floor (default)
        sigma2_max = 20.0;
        sigma2new  = max(sigma2new, sigma2_min);
        sigma2new  = min(sigma2new, sigma2_max);

        [Jmean, ~, x_seg] = sampleTrajectoryCosts(x0_seg, xd_seg, Unew, ...
                                                   Q, Qf, R_ctrl, dt, N, DYNAMICS);

        sigma2_min_history(i) = min(sigma2new);
        J_history(i)          = Jmean;

        fprintf('  Iter %3d | J_mean = %.4f | sigma2 min/mean/max = %.2e / %.2e / %.2e\n', ...
                i, Jmean, min(sigma2new), mean(sigma2new), max(sigma2new));

        U      = Unew;
        sigma2 = sigma2new;
    end

    % --- Plot convergence for this segment ---
    figure;
    subplot(2,1,1);
    semilogy(1:iterations, J_history, 'b-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Cost J');
    title(sprintf('Segment %d->%d: Cost convergence', seg, seg+1));
    grid on;

    subplot(2,1,2);
    semilogy(1:iterations, sigma2_min_history, 'r-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('min(sigma^2)');
    title(sprintf('Segment %d->%d: Min sigma^2 convergence', seg, seg+1));
    grid on;

    % --- Accumulate trajectory (skip first state to avoid duplicates) ---
    x_full = [x_full, x_seg(:, 2:end)];
    u_full = [u_full, U];

    % Record where next segment starts in x_full
    seg_breaks(end+1) = size(x_full, 2);

    % --- Warm-start next segment ---
    U_next = [U(:, 2:end), zeros(nU, 1)];
    if seg < (num_waypoints - 1)
        N_next = seg_params{seg+1}(1);
        if N_next > N
            U_next = [U_next, zeros(nU, N_next - N)];
        elseif N_next < N
            U_next = U_next(:, 1:N_next);
        end
        U = U_next;
    end

end

fprintf('\nFinal state:\n');
disp(x_full(:, end));
disp('Optimisation complete.');

% --- Save trajectory and controls to .mat ---
save('vtol_trajectory.mat', 'x_full', 'u_full', 'seg_breaks', 'waypoints', 'seg_params');
fprintf('Trajectory saved to vtol_trajectory.mat\n');

% -----------------------------------------------------------------------
% --- Plot full trajectory ---
% -----------------------------------------------------------------------
wp_pos = zeros(3, num_waypoints);
for k = 1:num_waypoints
    wp_pos(:, k) = waypoints{k}(1:3);
end

colors = lines(num_waypoints - 1);  % one color per segment

figure('Name', 'Full VTOL Trajectory', 'NumberTitle', 'off');

% --- 3D trajectory ---
ax3d = subplot(2, 2, [1 3]);
hold on; grid on; axis equal;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Trajectory');
view(45, 30);

for seg = 1:(num_waypoints - 1)
    idx_start = seg_breaks(seg);
    idx_end   = seg_breaks(seg + 1);
    seg_x = x_full(1, idx_start:idx_end);
    seg_y = x_full(2, idx_start:idx_end);
    seg_z = x_full(3, idx_start:idx_end);
    plot3(seg_x, seg_y, seg_z, '-', 'Color', colors(seg,:), 'LineWidth', 2, ...
          'DisplayName', sprintf('Seg %d->%d', seg, seg+1));
end

% Plot waypoints
scatter3(wp_pos(1,:), wp_pos(2,:), wp_pos(3,:), 80, 'k', 'filled', ...
         'DisplayName', 'Waypoints');
for k = 1:num_waypoints
    text(wp_pos(1,k), wp_pos(2,k), wp_pos(3,k) + 0.3, sprintf('  x%d', k-1), ...
         'FontSize', 9, 'FontWeight', 'bold');
end
legend('Location', 'best');

% --- XY projection ---
subplot(2, 2, 2);
hold on; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('Top View (XY)');
for seg = 1:(num_waypoints - 1)
    idx_start = seg_breaks(seg);
    idx_end   = seg_breaks(seg + 1);
    plot(x_full(1, idx_start:idx_end), x_full(2, idx_start:idx_end), ...
         '-', 'Color', colors(seg,:), 'LineWidth', 2);
end
scatter(wp_pos(1,:), wp_pos(2,:), 80, 'k', 'filled');
for k = 1:num_waypoints
    text(wp_pos(1,k)+0.2, wp_pos(2,k)+0.2, sprintf('x%d', k-1), 'FontSize', 8);
end

% --- XZ projection (altitude profile) ---
subplot(2, 2, 4);
hold on; grid on;
xlabel('X (m)'); ylabel('Z (m)');
title('Side View (XZ)');
for seg = 1:(num_waypoints - 1)
    idx_start = seg_breaks(seg);
    idx_end   = seg_breaks(seg + 1);
    plot(x_full(1, idx_start:idx_end), x_full(3, idx_start:idx_end), ...
         '-', 'Color', colors(seg,:), 'LineWidth', 2);
end
scatter(wp_pos(1,:), wp_pos(3,:), 80, 'k', 'filled');
for k = 1:num_waypoints
    text(wp_pos(1,k)+0.2, wp_pos(3,k)+0.2, sprintf('x%d', k-1), 'FontSize', 8);
end

sgtitle('VTOL Waypoint Trajectory (CEM)', 'FontSize', 13, 'FontWeight', 'bold');

% --- Control inputs over time ---
figure('Name', 'Control Inputs', 'NumberTitle', 'off');
u_labels = {'u_1', 'u_2', 'u_3', 'u_4'};
T_u = size(u_full, 2);
for k = 1:nU
    subplot(nU, 1, k);
    plot(1:T_u, u_full(k, :), 'LineWidth', 1.2);
    ylabel(u_labels{k});
    grid on;
    % Draw segment boundaries
    for sb = 2:length(seg_breaks)-1
        xline(seg_breaks(sb), '--k', 'Alpha', 0.4);
    end
    if k == 1
        title('Control Inputs (dashed lines = segment boundaries)');
    end
end
xlabel('Time step');

% -----------------------------------------------------------------------
function Qmat = buildDiagCost(w_pos, w_vel, w_quat, w_omega)
% Helper: build a 13x13 block-diagonal cost matrix from weight vectors.
% Inputs are vectors of weights for each block of the error vector
% [pos(3); vel(3); q_err(4); omega(3)].
Qmat = blkdiag(diag(w_pos), diag(w_vel), diag(w_quat), diag(w_omega));
end

% -----------------------------------------------------------------------
function [J, Jk, x] = sampleTrajectoryCosts(x0, xd, u, Q, Qf, R, dt, N, DYNAMICS)
x      = zeros(length(x0), N+1);
x(:,1) = x0;
Jk     = zeros(1, N+1);

for k = 1:N
    Jk(k)    = runningCost(x(:,k), xd, u(:,k), Q, R, dt);
    x(:,k+1) = x(:,k) + DYNAMICS(x(:,k), u(:,k)) * dt;
    x(7:10, k+1) = x(7:10, k+1) / norm(x(7:10, k+1));
end
Jk(N+1) = finalCost(x(:,N+1), xd, Qf);
J = sum(Jk);
end

% -----------------------------------------------------------------------
function J = runningCost(x, xd, u, Q, R, dt)
e = stateError(x, xd);
J = e'*Q*dt*e + u'*R*dt*u;
end

% -----------------------------------------------------------------------
function J = finalCost(xT, xd, Qf)
e = stateError(xT, xd);
J = e' * Qf * e;
end

% -----------------------------------------------------------------------
function e = stateError(x, xd)
e_pos   = x(1:3)   - xd(1:3);
e_vel   = x(4:6)   - xd(4:6);
e_omega = x(11:13) - xd(11:13);

q  = x(7:10)  / norm(x(7:10));
qd = xd(7:10) / norm(xd(7:10));

qd_conj = [qd(1); -qd(2:4)];
q_err   = quatmultiply_col(qd_conj, q);

if q_err(1) < 0
    q_err = -q_err;
end

e = [e_pos; e_vel; q_err; e_omega];
end

% -----------------------------------------------------------------------
function q_out = quatmultiply_col(a, b)
aw=a(1); ax=a(2); ay=a(3); az=a(4);
bw=b(1); bx=b(2); by=b(3); bz=b(4);
q_out = [aw*bw - ax*bx - ay*by - az*bz;
         aw*bx + ax*bw + ay*bz - az*by;
         aw*by - ax*bz + ay*bw + az*bx;
         aw*bz + ax*by - ay*bx + az*bw];
end