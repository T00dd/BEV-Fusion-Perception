system("python3 main.py");

data = load('cones.mat');

cones_left = data.cones_left;
cones_right = data.cones_right;

figure;
plot(cones_left(:,1), cones_left(:,2), 'b.', 'MarkerSize', 10); 
hold on;
plot(cones_right(:,1), cones_right(:,2), 'r.', 'MarkerSize', 10);

axis equal;
grid on;
xlabel('X [m]');
ylabel('Y [m]');
title('Track');
legend('Left cones', 'Right cones');

input('Press Enter to close the plot...');