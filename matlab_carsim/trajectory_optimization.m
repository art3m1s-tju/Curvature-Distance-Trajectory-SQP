% Global Minimum Time Trajectory Planning for Track Racing
% CarSim Integration Version
% 
% This script reproduces the algorithm of the paper to calculate the optimal
% trajectory balancing curvature and distance. It is written in a format
% that can be easily fed into CarSim's trajectory tracking controller.

clc; clear; close all;

%% 1. Parameters and Settings
% Vehicle width geometry and safety clearance
wv = 1.0; % Vehicle width constraint (half distance for reference point)
ws = 0.5; % Safety margin

% Weight coefficient between curvature and distance
% epsilon = 0    => Minimum curvature path
% epsilon = 10.0 => Balanced path
epsilon = 10.0;
max_iter = 5;

%% 2. Load Track Data
% Reads the boundary points of the track
fprintf('Loading track data...\n');
opts = detectImportOptions('../track.csv');
T = readtable('../track.csv', opts);

% Inner boundary p, outer boundary q
px = T.left_border_x;
py = T.left_border_y;
p = [px, py];

qx = T.right_border_x;
qy = T.right_border_y;
q = [qx, qy];

% Direction vector from inner to outer boundary
v = q - p;
vx = v(:, 1);
vy = v(:, 2);
N = size(p, 1);

%% 3. Create Constraints
% Calculate local track width at each point
road_widths = vecnorm(v, 2, 2);

% Safety boundaries for alpha (our decision variable)
alpha_min = (wv + ws) ./ road_widths;
alpha_max = 1.0 - (wv + ws) ./ road_widths;

% Cap alpha arrays to ensure feasibility at very narrow spots
alpha_min = max(alpha_min, 0.05);
alpha_max = min(alpha_max, 0.95);

% Inequality Constraints for 'quadprog'
% We need alpha_min <= alpha <= alpha_max
lb = alpha_min;
ub = alpha_max;

%% 4. Matrix Constructions
fprintf('Building structural matrices...\n');
% A: Difference Matrix (Sparse, First order)
e = ones(N, 1);
A = spdiags([-e, e], [0, 1], N, N);
A(N, 1) = 1; % Loop closure
ATA = A' * A;

% Distance Factor Hessian (Hs is constant during iterations)
Vx = spdiags(vx, 0, N, N);
Vy = spdiags(vy, 0, N, N);
Hs = 2 * (Vx' * ATA * Vx + Vy' * ATA * Vy);
fs = 2 * (Vx' * ATA * px + Vy' * ATA * py);

%% 5. SQP Iteration Loop
alpha_ref = 0.5 * ones(N, 1); % Initial guess (center of the track)
options = optimoptions('quadprog', 'Display', 'off');

for iter = 1:max_iter
    % 5.1 Reconstruct current reference trajectory
    rx = px + vx .* alpha_ref;
    ry = py + vy .* alpha_ref;
    
    % 5.2 Derivatives and Weights calculation
    rx_prime = circshift(rx, -1) - rx;
    ry_prime = circshift(ry, -1) - ry;
    
    ds = sqrt(rx_prime.^2 + ry_prime.^2);
    rx_prime = rx_prime ./ (ds + 1e-6);
    ry_prime = ry_prime ./ (ds + 1e-6);
    
    % Formulation of Curvature weighting matrix T
    denom = (rx_prime.^2 + ry_prime.^2).^1.5 + 1e-8;
    Txx_diag = (ry_prime.^2) ./ denom;
    Tyy_diag = (rx_prime.^2) ./ denom;
    Txy_diag = -(rx_prime .* ry_prime) ./ denom;
    
    Txx = spdiags(Txx_diag, 0, N, N);
    Tyy = spdiags(Tyy_diag, 0, N, N);
    Txy = spdiags(Txy_diag, 0, N, N);
    
    % 5.3 Spline 2nd order approximation Matrix M
    M = spdiags([e, -2*e, e], [-1, 0, 1], N, N);
    M(1, N) = 1;
    M(N, 1) = 1;
    ds_diag = spdiags(1.0 ./ (ds.^2 + 1e-6), 0, N, N);
    M = ds_diag * M;
    
    % Calculates composite parts of Hk
    MT_Txx_M = M' * Txx * M;
    MT_Tyy_M = M' * Tyy * M;
    MT_Txy_M = M' * Txy * M;
    
    Hk = 2 * (Vx'*MT_Txx_M*Vx + Vy'*MT_Tyy_M*Vy + Vx'*MT_Txy_M*Vy + Vy'*MT_Txy_M*Vx);
    fk = 2 * (Vx'*MT_Txx_M*px + Vy'*MT_Tyy_M*py + Vx'*MT_Txy_M*py + Vy'*MT_Txy_M*px);
    
    % 5.4 Build joint Objective Formulation
    % Combines curvature factors and distance factors adjusted by epsilon
    H = Hk + epsilon * Hs;
    f = fk + epsilon * fs;
    
    % Symmetrize H to avoid MATLAB numerical warning
    H = (H + H') / 2;
    
    % 5.5 QP Solver execution
    alpha_new = quadprog(H, f, [], [], [], [], lb, ub, alpha_ref, options);
    
    if isempty(alpha_new)
        warning('Solver failed at iteration %d.', iter);
        break;
    end
    
    diff = max(abs(alpha_new - alpha_ref));
    fprintf('Iteration %d: max alpha adjustment = %f\n', iter, diff);
    alpha_ref = alpha_new;
    
    if diff < 1e-4
        fprintf('Algorithm Converged.\n');
        break;
    end
end

%% 6. Final Results and Data Export for CarSim
optimal_x = px + vx .* alpha_ref;
optimal_y = py + vy .* alpha_ref;

% Save out to a space-delimited text or csv, typical for CarSim parsers
carsim_path_data = [optimal_x, optimal_y];
writematrix(carsim_path_data, 'optimal_trajectory_for_carsim.csv');
fprintf('Success! Saved CarSim compatible trajectory to: optimal_trajectory_for_carsim.csv\n');

%% 7. Visualization
figure('Name', 'Optimal Path Plot', 'NumberTitle', 'on', 'Position', [100, 100, 800, 800]);
hold on; grid on; axis equal;
% Draw track boundaries
plot(px, py, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Inner Boundary');
plot(qx, qy, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Outer Boundary');

% Overlay target trajectory
plot(optimal_x, optimal_y, 'b-.', 'LineWidth', 2, 'DisplayName', sprintf('Optimal Trajectory (eps=%g)', epsilon));

% Set viewing range limit to visualize part of track clearly
xlim([min(px)-20, max(px)+20]);
ylim([min(py)-20, max(py)+20]);
legend('Location', 'best');
xlabel('X Position [m]');
ylabel('Y Position [m]');
title('Calculated Trajectory Path Segment');
hold off;
