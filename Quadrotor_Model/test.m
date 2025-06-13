model = @quadload;

x0 = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]';
u =  [0.7*9.81;0;0;0];
dt = 0.1;
x = x0;

for i = 1:5
    x = x + model(x, u)*dt;
    x
end
