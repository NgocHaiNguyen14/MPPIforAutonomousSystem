%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw function for autonomous vehicle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function draw_vehicle(t, x, param, traj)
    x_pos = x(1);
    y_pos = x(2);
    psi = x(3);

    w = 1.8;
    l = 4;
    vertices = [l/2, w/2; l/2, -w/2; -l/2, -w/2; -l/2, w/2];
    R = [cos(psi), -sin(psi); sin(psi), cos(psi)];
    vertices_rot = (R * vertices')';
    vertices_trans = vertices_rot + [x_pos, y_pos];

    figure(25); cla; hold on;
    try
        grass_vertices = [-10, -12; 50, -12; 50, 12; -10, 12];
        fill(grass_vertices(:,1), grass_vertices(:,2), [0 0.6 0], 'FaceAlpha', 0.3);
        x_road = linspace(-10, 50, 100);
        road_vertices = [x_road', param.road_width/2*ones(100,1); ...
                         flipud([x_road', -param.road_width/2*ones(100,1)])];
        fill(road_vertices(:,1), road_vertices(:,2), [0.5 0.5 0.5], 'FaceAlpha', 0.7);
        plot(x_road, zeros(size(x_road)), 'w--', 'LineWidth', 1.5);
        theta = linspace(0, 2*pi, 100);
        % Moving vehicles
        mv_positions = update_moving_vehicles(param, t);
        for m = 1:length(param.moving_vehicles)
            l_mv = param.moving_vehicles(m).length;
            w_mv = param.moving_vehicles(m).width;
            vertices_mv = [l_mv/2, w_mv/2; l_mv/2, -w_mv/2; -l_mv/2, -w_mv/2; -l_mv/2, w_mv/2];
            vertices_mv_trans = vertices_mv + [mv_positions(m).x, mv_positions(m).y];
            fill(vertices_mv_trans(:,1), vertices_mv_trans(:,2), 'g', 'FaceAlpha', 0.5);
        end
        % Trees
        rng(0);
        num_trees = 20;
        tree_x = [rand(num_trees/2,1)*60-10; rand(num_trees/2,1)*60-10];
        tree_y = [rand(num_trees/2,1)*5+param.road_width/2; rand(num_trees/2,1)*5-param.road_width/2-5];
        for i = 1:num_trees
            tree_circle = 0.3 * cos(theta) + tree_x(i);
            tree_circle_y = 0.3 * sin(theta) + tree_y(i);
            plot(tree_circle, tree_circle_y, 'b-', 'LineWidth', 1);
        end
        % Autonomous vehicle
        if all(isfinite(x))
            fill(vertices_trans(:,1), vertices_trans(:,2), 'b', 'FaceAlpha', 0.5);
            plot(x_pos, y_pos, 'b.', 'MarkerSize', 10);
        end
        if ~isempty(traj) && all(isfinite(traj(:)))
            plot(traj(1,:), traj(2,:), 'k-', 'LineWidth', 1);
        end
        axis equal; axis([-10 50 -12 12]);
        title(['Time: ', num2str(t, '%.2f'), ' s']);
        xlabel('x (m)'); ylabel('y (m)');
    catch e
        fprintf('Plotting error at t=%.2f: %s\n', t, e.message);
    end
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update positions of moving vehicles (duplicate for draw_vehicle)
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