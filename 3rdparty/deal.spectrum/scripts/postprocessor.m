%%
%close all
%clear all
%clc

%% settings
dim  = 2;
N    = 64;
bins = 1;

%% init flow field

if dim==2
    % ... uniform grid
    x = linspace(0,2*pi,N+1); x = x(1:end-1);
    [X,Y] = meshgrid(x,x);

    % ... u:
    u = cos(1*X).*cos(2*Y);
    % ... v:
    v = cos(4*X).*cos(3*Y);
else
    % ... uniform grid
    x = linspace(0,2*pi,N+1); x = x(1:end-1);
    [X,Y,Z] = meshgrid(x,x,x);

    % ... u:
    u = cos(1*X);
    % ... v:
    v = cos(2*Y);
    % ... w:
    w = cos(4*Z);
end

%% perform transform
% ... scaling factor
scaling = (2*pi*N)^dim;

% ... FFT
NFFT = 2.^nextpow2(size(u));
u_fft=fftn(u,NFFT)./scaling;
v_fft=fftn(v,NFFT)./scaling;

% ...
phi = 0.5 * (u_fft.*conj(u_fft) + v_fft.*conj(v_fft));

if dim == 3
    w_fft=fftn(w,NFFT)./scaling;
    phi = phi + 0.5 * w_fft.*conj(w_fft);
end

%% Perform postprocessing
n = N;

% ... allocate memory
e = zeros(n,1); l = zeros(n,1); c = zeros(n,1);

if dim == 2
    % ... loop over all (kappa_1, kappa_2)
    for j=1:N
        for i=1:N
            % ... radius of kappa vector
            r    = sqrt(min(i-1,N+1-i)^2+min(j-1,N+1-j)^2);
            % ... index within result vectors
            p    = round(bins*r)+1;
            % ... local energy
            e(p) = e(p) + phi(i,j);
            % ... radius for averaging
            l(p) = l(p) + r;
            % ... counter
            c(p) = c(p) + 1;
        end 
    end
else
    % ... loop over all (kappa_1, kappa_2, kappa_3)
    for k=1:N
        for j=1:N
            for i=1:N
                % ... radius of kappa vector
                r    = sqrt(min(i-1,N+1-i)^2+min(j-1,N+1-j)^2+min(k-1,N+1-k)^2);
                % ... index within result vectors
                p    = round(r+0.0)+1;
                % ... local energy
                e(p) = e(p) + phi(i,j,k);
                % ... radius for averaging
                l(p) = l(p) + r;
                % ... counter
                c(p) = c(p) + 1;
            end 
        end
    end
end

% normalize by count: wave-number...
l = l./c; 
% ... local energy
e = e./c;

% spectral energy: integrate energy over surface
if dim == 2
    e = 2 * pi * e .* l;
else
    e = 4 * pi * e .* l.^2;
end

%% collect results
res = [l,c,e];
res = res(c>0 & e>1e-15,:);

% ... and print as table
kappa = res(:,1); count = res(:,2); energy = res(:,3);
T = table(kappa,count,energy)