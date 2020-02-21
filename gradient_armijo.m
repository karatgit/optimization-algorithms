% Copyright: Theocharis Karathymios (github name: karatgit)
% Minimization of function f using the Gradient method with Armijo Line Search
% Define the objective function
syms x y
f(x,y) = 100*(y-x.^2).^2 + (1-x).^2;
% Calculate gradient of f
grad_f(x,y) = gradient(f);
% Set initial point and Armijo constants
x_k = -3/4;
y_k = 1;
g = 0.5;
s = 0.5;
a = 1;
epsilon = 0.01;
% Calculate f(xk)
f_xk = double(f(x_k,y_k));
% gradf(xk)
gradf_xk = double(grad_f(x_k,y_k));
% Set dk
dk = -gradf_xk;
% f(x+a*dk)
f_adk = double(f(x_k+a*dk(1,1),y_k+a*dk(2,1)));
% Starting point for later storage of x,y
X = x_k;
Y = y_k;
J = log((x_k-1)^2+(y_k-1)^2);
%________Gradient Algorithm_______
while norm(dk)>epsilon
    %_________Armijo step_________
    a = 1;
    while f_adk - f_xk > g*a*gradf_xk'*dk
    a = a*s;
    f_adk = double(f(x_k+a*dk(1,1),y_k+a*dk(2,1)));
    end
    %_____________________________
    x_k = x_k + a*dk(1,1);
    y_k = y_k + a*dk(2,1);
    J_new = log((x_k-1)^2+(y_k-1)^2);
    X = [X x_k];
    Y = [Y y_k];
    J = [J J_new];
    gradf_xk = double(grad_f(x_k,y_k));
    dk = -gradf_xk;
    f_xk = double(f(x_k,y_k));
    f_adk = double(f(x_k+a*dk(1,1),y_k+a*dk(2,1)));
end
optimal_point = [x_k;y_k];
fcontour(f);
hold on;
plot(X,Y);
hold off;
plot(J);