%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation of an autonomous vehicle navigating a road with obstacles
% using MPPI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vehicle_mppi
    close all
    clear all

    addpath("dynamical_systems/")
    param = init_vehicle_params();
    param.k_lane = 1000; % Increased lane boundary cost
    DYNAMICS = @vehicle_dynamics;
    DRAW = @draw_vehicle;
    rmpath("dynamical_systems/")

    T = 5.0; % Increased time horizon for longer road
    dt = 0.05; % time step
    N = floor(T/dt);
    nX = 4; % states: [x, y, psi, v]
    nU = 2; % inputs: [delta, a]

    t = zeros(1,N);
    x = zeros(nX,N);
    u = zeros(nU,N);

    % Cost matrices
    Q = diag([1, 100, 1, 1]); % Increased y-position penalty
    R = 0.1 * eye(nU);
    Qf = diag([200, 1000, 10, 10]); % Increased x, y penalties

    % Desired state
    xd = zeros(nX,1);
    xd(1) = 90; % End of the longer road
    xd(2) = 0;  % Middle of the lane
    xd(3) = 0;  % Aligned heading
    xd(4) = param.v_des;
    param.xd = xd;

    % Initial state
    x(:,1) = [0; 0; 0; param.v_des]; % Start at centerline

    % Run MPPI
    U = MPPI(Q, R, Qf, x(:,1), xd, N, dt, param, DYNAMICS);

    % Simulate
    traj = zeros(2,N);
    for k = 1:N-1
        x(:,k+1) = x(:,k) + DYNAMICS(t(k), x(:,k), U(:,k), param) * dt;
        t(k+1) = t(k) + dt;
        traj(:,k) = x(1:2,k);
        if any(isnan(x(:,k+1)) | isinf(x(:,k+1)))
            fprintf('Invalid state at k=%d: %s\n', k+1, mat2str(x(:,k+1)));
            break;
        end
    end
    traj(:,N) = x(1:2,N);

    % Save debug data
    save('debug_states.mat', 'x', 'u');

    % Animate
    figure(25); set(gcf, 'Position', [100, 100, 800, 600]);
    for k = 1:N
        DRAW(t(k), x(:,k), param, traj(:,1:k));
        pause(0.05);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MPPI for vehicle control
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function U = MPPI(Q, R, Qf, x0, xd, N, dt, param, DYNAMICS)
    num_samples = 10; % Increased for better control
    iterations = 100; % Increased for better optimization
    lambda = 50;
    nu = 0.005;
    rho = 10000;

    Jsave = [];
    ksave = [];

    U = zeros(length(R), N);
    u = zeros(length(R), N);
    x = zeros(length(xd), N);

    figure(7); set(gcf, 'Position', [900, 100, 800, 600]);
    for i = 1:iterations
        fprintf('Iteration %d\n', i);
        figure(7)
        cla;
        J = [];
        weights = [];
        dU = cell(num_samples,1);
        valid_samples = 0;
        for j = 1:num_samples
            x(:,1) = x0;
            du = sqrt(nu) * randn(length(R), N) / (sqrt(rho) * sqrt(dt));
            dU{j} = du;
            u = U + du;
            t = 0;
            [~, Jk, x] = sampleTrajectoryCosts(x0, xd, u, Q, R, Qf, dt, N, param, DYNAMICS);
            if any(isnan(x(:)) | isinf(x(:)))
                fprintf('Invalid state in sample %d, iteration %d\n', j, i);
                continue;
            end
            Ji = getCostToGo(Jk);
            if isnan(Ji) || isinf(Ji)
                fprintf('Invalid cost in sample %d, iteration %d: %f\n', j, i, Ji);
                continue;
            end
            valid_samples = valid_samples + 1;
            J = [J; Ji];
            weights = [weights; Ji];
            figure(7)
            hold on
            if all(isfinite(x(:)))
                plot(x(1,:), x(2,:),'b','LineWidth',0.5)
            end
            hold off
            pause(0.01);
        end
        if valid_samples == 0
            fprintf('No valid samples in iteration %d, using last U\n', i);
            break;
        end
        draw_environment(param, i * dt * N);
        % Normalize weights
        J = J - min(J);
        expS = exp(-J/lambda);
        expS = expS / (sum(expS) + eps);
        % Update control
        for j = 1:N
            su = 0;
            for k = 1:valid_samples
                su = su + (expS(k) * dU{k}(:,j));
            end
            U(:,j) = U(:,j) + su;
            if any(isnan(U(:,j)) | isinf(U(:,j)))
                fprintf('Invalid control at j=%d, iteration %d: %s\n', j, i, mat2str(U(:,j)));
                U(:,j) = zeros(length(R),1);
            end
        end
        u = U;
        [~, Jk, x] = sampleTrajectoryCosts(x0, xd, u, Q, R, Qf, dt, N, param, DYNAMICS);
        Jc = sum(Jk);
        Jsave = [Jsave Jc];
        ksave = [ksave i];
    end

    figure(8); set(gcf, 'Position', [100, 700, 800, 300]);
    plot(ksave, Jsave)
    xlabel('Iterations')
    ylabel('Cost')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample trajectory costs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J, Jk, x] = sampleTrajectoryCosts(x0, xd, u, Q, R, Qf, dt, N, param, DYNAMICS)
    x(:,1) = x0;
    t = 0;
    fprintf('Entering sampleTrajectoryCosts\n'); % Debug statement
    Jk = zeros(1,N+1);
    for k = 1:N
        obst_cost = 0;
        % Static obstacles removed
        % for o = 1:length(param.obstacles)
        %     dx = x(1,k) - param.obstacles(o).x;
        %     dy = x(2,k) - param.obstacles(o).y;
        %     dist_to_obst = max(sqrt(dx^2 + dy^2), 0.1);
        %     obst_cost = obst_cost + min(param.k_obst * exp(-dist_to_obst / param.sigma_obst), 1e4);
        % end
        % Dynamic obstacles (moving vehicles)
        mv_positions = update_moving_vehicles(param, t);
        for m = 1:length(param.moving_vehicles)
            dx_mv = x(1,k) - mv_positions(m).x;
            dy_mv = x(2,k) - mv_positions(m).y;
            dist_to_mv = max(sqrt(dx_mv^2 + dy_mv^2), 0.1);
            obst_cost = obst_cost + min(param.k_obst * exp(-dist_to_mv / param.sigma_obst), 1e4);
        end
        lane_bound_cost = param.k_lane * (max(0, abs(x(2,k)) - param.road_width/2)^2 + 0.1 * x(2,k)^2);
        state_cost = min(x(:,k)'*Q*dt*x(:,k), 1e4);
        control_cost = min(u(:,k)'*R*dt*u(:,k), 1e4);
        Jk(k) = state_cost + control_cost + obst_cost + lane_bound_cost;
        x(:,k+1) = x(:,k) + DYNAMICS(t, x(:,k), u(:,k), param) * dt;
        t = t + dt;
    end
    Jk(N+1) = min((x(:,N) - xd)'*Qf*(x(:,N) - xd), 1e4);
    J = sum(Jk);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cost-to-go
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S = getCostToGo(Jk)
    S = sum(Jk);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update positions of moving vehicles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mv_positions = update_moving_vehicles(param, t)
    mv_positions = struct('x', {}, 'y', {});
    for m = 1:length(param.moving_vehicles)
        x = param.moving_vehicles(m).x0 + param.moving_vehicles(m).vx * t;
        y = param.moving_vehicles(m).y0;
        mv_positions(m).x = x;
        mv_positions(m).y = y;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw environment with dynamic obstacles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function draw_environment(param, t)
    hold on;
    grass_vertices = [-10, -15; 90, -15; 90, 15; -10, 15];
    fill(grass_vertices(:,1), grass_vertices(:,2), [0 0.6 0], 'FaceAlpha', 0.3);
    x_road = linspace(-10, 90, 100);
    road_vertices = [x_road', param.road_width/2*ones(100,1); ...
                     flipud([x_road', -param.road_width/2*ones(100,1)])];
    fill(road_vertices(:,1), road_vertices(:,2), [0.5 0.5 0.5], 'FaceAlpha', 0.7);
    plot(x_road, zeros(size(x_road)), 'w--', 'LineWidth', 1.5);
    theta = linspace(0, 2*pi, 100);
    % Static obstacles removed
    % for o = 1:length(param.obstacles)
    %     x_obst = param.obstacles(o).x + param.obstacles(o).r * cos(theta);
    %     y_obst = param.obstacles(o).y + param.obstacles(o).r * sin(theta);
    %     plot(x_obst, y_obst, 'r-', 'LineWidth', 2);
    % end
    % Moving vehicles
    mv_positions = update_moving_vehicles(param, t);
    for m = 1:length(param.moving_vehicles)
        l = param.moving_vehicles(m).length;
        w = param.moving_vehicles(m).width;
        vertices = [l/2, w/2; l/2, -w/2; -l/2, -w/2; -l/2, w/2];
        vertices_trans = vertices + [mv_positions(m).x, mv_positions(m).y];
        fill(vertices_trans(:,1), vertices_trans(:,2), 'g', 'FaceAlpha', 0.5);
    end
    % Trees
    rng(0);
    num_trees = 20;
    tree_x = [rand(num_trees/2,1)*100-10; rand(num_trees/2,1)*100-10];
    tree_y = [rand(num_trees/2,1)*5+param.road_width/2; rand(num_trees/2,1)*5-param.road_width/2-5];
    for i = 1:num_trees
        tree_circle = 0.3 * cos(theta) + tree_x(i);
        tree_circle_y = 0.3 * sin(theta) + tree_y(i);
        plot(tree_circle, tree_circle_y, 'b-', 'LineWidth', 1);
    end
    axis equal; axis([-10 90 -15 15]);
    hold off;
end