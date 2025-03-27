t = 0.0:0.1:10; % time

h = 0.9 * (t >= 0 & t <= 10); % h(t)
x = (t>=2); % x(t)

y = conv(x, h, 'same')*0.01; % convolution y(t)

figure;
plot(t, y)
title("Convolution of Signals");

t = linspace(0,10,1000); % time

k = 0:11;

x_k = zeros(length(k), length(t));

for i = 1:length(k)
    x_k(i, :) = ((-1).^k(i) ./ ((k(i)*pi) .^ 2) .* exp(1j*k(i) * pi * t));
end

x_t = 1/2 + sum(x_k, 1);

figure;
subplot(2,1,1);
stem(k, abs(x_k(:,1)), 'filled');
title("Magnitude Spectrum");

subplot(2,1,2);
stem(k, angle(x_k(:,1)), 'filled');
title("Phase Spectrum");

T_values = [1, 0.1, 0.01]; % periods

figure;
hold on;

for T=T_values
    t= 0:T:10;
    x_t = 2*(t>=2) - (t>=2);
    v_t = 2 - t .* (t>=2);

    y = conv(x_t, v_t, 'same') * T; % convolution

    plot(t,y, 'DisplayName', ['T= ' num2str(T)]);
end
title("Effect of Sampling Period");
legend('show');