% Ryan DuBrueler

%% PART 1 - Raised Cosine Signal
% Define the time vector from 0 to 4 with increments of 0.01
t = 0:0.01:4;

% Define the signal x(t) = (1 + cos(pi*(t-2))) * rect(0,2)
x1 = (1 + cos(pi*(t-2))) .* rectangularPulse(0,2,t);

% Plot the signal
figure(1);
subplot(3,2,1)
plot(t, x1)
xlabel('t')
ylabel('x(t)')
title('Signal x(t) = (1 + cos(\pi (t-2)))*rect(t-2)')
grid on

% Define the angular frequency vector
w = -12:0.01:12; 

X = zeros(size(w));

% Compute the CTFT numerically via the trapezoidal rule.
for k = 1:length(w)
    % Create the integrand for the current omega:
    % x(t) * exp(-j*w(k)*t)
    integrand = x1 .* exp(-1j * w(k) * t);
    
    % Numerically integrate using trapz.
    X(k) = trapz(t, integrand);
end

% Plot the magnitude and phase of the Fourier transform.
subplot(3,2,3)
plot(w, abs(X))
xlabel('\omega [rad/s]')
ylabel('|X(j\omega)|')
title('Magnitude')
grid on

subplot(3,2,4)
plot(w, angle(X))
xlabel('\omega [rad/s]')
ylabel('Phase (radians)')
title('Phase')
grid on

% Compute the inverse Fourier transform numerically using the inverse integral
% The inverse Fourier transform is given by:
%    x(t) = (1/(2π)) ∫ X(jω)e^(jωt) dω
% Define a reconstruction time vector for which we wish to recover x(t).
t_recon = 0:0.01:4;           
x_recon = zeros(size(t_recon));
x_recon = real(x_recon);

% For each time value, evaluate the inverse integral using trapz.
for i = 1:length(t_recon)
    % For current time, compute the product X(jω)e^(jωt)
    integrand = X .* exp(1j*w*t_recon(i));
    x_recon(i) = trapz(w, integrand) / (2*pi);
end

% Plot original signal (computed over t) for comparison
subplot(3,2,5)
plot(t, x1)
xlabel('Time t (seconds)')
ylabel('x(t)')
title('Original Signal x(t)')
grid on

% Plot the recovered signal (using the inverse Fourier transform)
subplot(3,2,6)
plot(t_recon, real(x_recon), 'r', 'LineWidth', 1.5)
xlabel('Time t (seconds)')
ylabel('x_{recovered}(t)')
title('Signal Recovered')
grid on
% Figure title
sgtitle('Part 1 - Raised Cosine Signal')
% Compute the energy in the time domain
% The energy in the time domain E_t is defined as:
%   E_t = ∫ |x(t)|^2 dt
E_time = trapz(t, abs(x1).^2);

% Compute the CTFT of x(t) numerically
% The CTFT is defined as:
%   X(jω) = ∫ x(t)e^(-jωt) dt,
% Define the frequency vector to make time-domain = frequency domain              
w = -100:.01:100;            

% Preallocate X for speed.
X = zeros(size(w));

% Compute the Fourier transform numerically (using trapezoidal integration)
for k = 1:length(w)
    integrand = x1 .* exp(-1j * w(k) * t);
    X(k) = trapz(t, integrand);
end

% Compute the energy in the frequency domain
% The energy in the frequency domain is given by:
%   E_f = (1/(2π)) ∫ |X(jω)|^2 dω
E_freq = trapz(w, abs(X).^2) / (2*pi);

% Display the computed energies
fprintf('Energy for x1(t) computed in time domain:  %f\n', E_time);
fprintf('Energy for x1(w) computed in frequency domain:  %f\n', E_freq);

%% Part 2 - Triangular Signal
% Define the time vector from 0 to 4 with increments of 0.01
t = 0:0.01:4;

% Define the signal x(t) = (1 + cos(pi*(t-2))) * rect(0,2)
x2 = (1 - abs(t-2)) .* rectangularPulse(0,2,t);

% Plot the signal
figure(2);
subplot(3,2,1)
plot(t, x2)
xlabel('t')
ylabel('x(t)')
title('Signal x_2(t) = (1-|t-2|)*rect(t-2)')
grid on

% Define the angular frequency vector
w = -12:0.01:12; 

X = zeros(size(w));

% Compute the CTFT numerically via the trapezoidal rule.
for k = 1:length(w)
    % Create the integrand for the current omega:
    % x(t) * exp(-j*w(k)*t)
    integrand = x2 .* exp(-1j * w(k) * t);
    
    % Numerically integrate using trapz.
    X(k) = trapz(t, integrand);
end

% Plot the magnitude and phase of the Fourier transform.
subplot(3,2,3)
plot(w, abs(X))
xlabel('\omega [rad/s]')
ylabel('|X(j\omega)|')
title('Magnitude')
grid on

subplot(3,2,4)
plot(w, angle(X))
xlabel('\omega [rad/s]')
ylabel('Phase (radians)')
title('Phase')
grid on

% Compute the inverse Fourier transform numerically using the inverse integral
% The inverse Fourier transform is given by:
%    x(t) = (1/(2π)) ∫ X(jω)e^(jωt) dω
% Define a reconstruction time vector for which we wish to recover x(t).
t_recon = 0:0.01:4;           
x_recon = zeros(size(t_recon));
x_recon = real(x_recon);

% For each time value, evaluate the inverse integral using trapz.
for i = 1:length(t_recon)
    % For current time, compute the product X(jω)e^(jωt)
    integrand = X .* exp(1j*w*t_recon(i));
    x_recon(i) = trapz(w, integrand) / (2*pi);
end

% Plot original signal (computed over t) for comparison
subplot(3,2,5)
plot(t, x2)
xlabel('Time t (seconds)')
ylabel('x(t)')
title('Original Signal x_2(t)')
grid on

% Plot the recovered signal (using the inverse Fourier transform)
subplot(3,2,6)
plot(t_recon, real(x_recon), 'r')
xlabel('Time t (seconds)')
ylabel('x_{recovered}(t)')
title('Signal Recovered')
grid on
% Figure title
sgtitle('Part 2 - Triangular Signal')
% Compute the energy in the time domain
% The energy in the time domain E_t is defined as:
%   E_t = ∫ |x(t)|^2 dt
E_time = trapz(t, abs(x2).^2);

% Compute the CTFT of x(t) numerically
% The CTFT is defined as:
%   X(jω) = ∫ x(t)e^(-jωt) dt,
% Define the frequency vector to make time-domain = frequency domain              
w = -100:.01:100;            

% Preallocate X for speed.
X = zeros(size(w));

% Compute the Fourier transform numerically (using trapezoidal integration)
for k = 1:length(w)
    integrand = x2 .* exp(-1j * w(k) * t);
    X(k) = trapz(t, integrand);
end

% Compute the energy in the frequency domain
% The energy in the frequency domain is given by:
%   E_f = (1/(2π)) ∫ |X(jω)|^2 dω
E_freq = trapz(w, abs(X).^2) / (2*pi);

% Display the computed energies
fprintf('Energy for x2(t) computed in time domain:  %f\n', E_time);
fprintf('Energy for x2(w) computed in frequency domain:  %f\n', E_freq);

%% Part 3 - Modulated Signal
% time vector
t = -10:.01:10;          
% x3(t) = (1+cos(pi*t))*sin^2(t)
x3 = (1 + cos(pi*t)) .* (sin(t).^2);
% Carrier: 10*cos(4pi*t)
carrier = 10 * cos(4*pi*t);
% Modulated signal: s(t)= x3(t).*carrier
s = x3 .* carrier;

% Compute FFT and shift zero-frequency to the center
X = fftshift(fft(x3)) * .01;
S = fftshift(fft(s)) * .01;

% frequency vectors
f = (-2:.002:2); 
omega = 2*pi.*f;    

% Plot x3(t)
figure(3);
subplot(2,2,1)
plot(t, x3)
xlabel('Time (s)')
ylabel('x_3(t)')
title('Signal x_3(t) = (1+cos(\pi t)) sin^2(t)')
grid on

% Plot s(t)
subplot(2,2,2)
plot(t, s)
xlabel('Time (s)')
ylabel('s(t)')
title('Modulated Signal s(t) = x_3(t)*10cos(4\pi t)')
grid on

% Plot |X(j\omega)|
subplot(2,2,3)
plot(omega, abs(X))
xlabel('\omega (rad/s)')
ylabel('|X(j\omega)|')
title('Magnitude Spectrum of X_3(j\omega)')
grid on

% Plot |S(j\omega)|
subplot(2,2,4)
plot(omega, abs(S))
xlabel('\omega (rad/s)')
ylabel('|S(j\omega)|')
title('Magnitude Spectrum of S(j\omega)')
grid on
% Figure title
sgtitle('Part 3 - Modulated Signal')
