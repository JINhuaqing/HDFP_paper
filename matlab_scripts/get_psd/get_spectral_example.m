% the signal
t = linspace(0, 1, 1000);
x = 1* sin(2*pi*10*t) + sin(2*pi*5*t);
%plot(t, x)

fmin = 1; %Hz
fmax = 40;
fs=1000;

[q, fsamples] = get_spectral(x, fs, 1000);
q_db = pow2db(q); % pow to db 
plot(fsamples, q_db);

x_deci = decimate(x, 8);
[q, fsamples] = get_spectral(x_deci, fs/8, 1000);
q_db = pow2db(q); % pow to db 
hold on
plot(fsamples, q_db);

plot(t, x);
t1 = linspace(0, 1, 1000/8);
hold on 
plot(t1, x_deci)