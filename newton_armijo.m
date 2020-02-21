% Copyright: Theocharis Karathymios (github name: karatgit)
% Minimization of function f using the Newton method with Armijo Line Search
% Define the objective function
syms x y
f(x,y) = 100*(y-x.^2).^2 + (1-x).^2;
% Calculate gradf
grad_f(x,y) = gradient(f);
grad2f(x,y) = hessian(f);
% Set initial point and Armijo constants
x_k = -3/4;
y_k = 1;
epsilon = 0.001;
a = 1;
g = 0.5;
sigma = 0.5;
% Set functions
% grad2f(xk)
grad2f_xk = double(grad2f(x_k,y_k));
grad2f_xk_inv = inv(grad2f_xk);
% f(xk)
f_xk = double(f(x_k,y_k));
% gradf(xk)
gradf_xk = double(grad_f(x_k,y_k));
%dk
dk = -gradf_xk;
% s
s = -grad2f_xk_inv*gradf_xk;
X =x_k;
Y = y_k;
J = log((x_k-1)^2+(y_k-1)^2);
% Newton's Algorithm
%_________________________________
while norm(gradf_xk)>epsilon
    grad2f_xk = double(grad2f(x_k,y_k));
    if det(grad2f_xk) < 1e-6
        dk = -gradf_xk;
    else
        s = -grad2f_xk_inv*gradf_xk;
        if abs(gradf_xk'*s) < epsilon*norm(gradf_xk)*norm(s)
            dk = -gradf_xk;
        else
            if gradf_xk'*s < 0
                dk =s;
            else
                if gradf_xk'*s > 0
                    dk = -s;
                end
            end
        end
    end
    % Armijo step
    %_____________________________
    f_adk = double(f(x_k+a*dk(1,1),y_k+a*dk(2,1)));
    a = 1;
    while f_adk - f_xk > g*a*gradf_xk'*dk
    a = a*sigma;
    f_adk = double(f(x_k+a*dk(1,1),y_k+a*dk(2,1)));
    end
    %_____________________________
    x_k = x_k + a*s(1,1);
    y_k = y_k + a*s(2,1);
    J_new = log((x_k-1)^2+(y_k-1)^2);
    gradf_xk = double(grad_f(x_k,y_k));
    grad2f_xk = double(grad2f(x_k,y_k));
    grad2f_xk_inv = inv(grad2f_xk);
    s = -grad2f_xk_inv*gradf_xk;
    X = [X x_k];
    Y = [Y y_k];
    J = [J J_new];
    dk = double(grad_f(x_k,y_k));
    f_xk = double(f(x_k,y_k));
    f_adk = double(f(x_k-a*dk(1,1),y_k-a*dk(2,1)));
end
optimal_point = [x_k;y_k];
fcontour(f);
hold on;
plot(X,Y);
hold off;
plot(J);
ylabel('Cost J');
xlabel('step');