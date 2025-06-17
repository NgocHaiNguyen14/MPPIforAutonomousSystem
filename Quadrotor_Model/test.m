model = @vtol2;

x0 = [0 0 0 0 0 0 0 0 0 0 0 0]';
u =  [10; 10; 0; 0];
dt = 0.1;
x = x0;

for i = 1:10
    x = x + model(x, u)*dt;
    x'
end
