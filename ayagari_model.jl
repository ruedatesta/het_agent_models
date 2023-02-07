# Ayagari model

using Optim, Interpolations, Plots


function discretize_assets(amin, amax, n_a)
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1 + log(1 + amax - amin));

    # make uniform grid
    u_grid = LinRange(0, ubar, n_a);
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1;
end

function rouwenhorst_Π(N, p)
    # base case Π_2
    Π = [p 1-p; 1-p p];

    # recursion to build up from Π_2 to Π_N
    for n in 3:N
        z = zeros(n-1,1);
        Π = p*[[Π z] ; [z' 0]] + (1-p)*[[z Π] ; [0 z']] + (1-p)*[[z' 0] ; [Π z]] + p*[[0 z'] ; [z Π]];
        Π[2:n-1,:] /= 2;
    end

    return(Π);
end

function stationary_markov(Π)
    Π_lim = Π^(1_000*size(Π,1)^2);
    Π_lim = Π_lim[1,:];
    Π_stat = Π_lim./sum(Π_lim);
    return(Π_stat);
end

function discretize_income(ρ, σ, n_s)
    # choose inner-switching probability p to match persistence rho
    p = (1+ρ)/2;

    # start with states from 0 to n_s-1, scale by alpha to match standard deviation sigma
    s = LinRange(0, n_s-1, n_s)
    alpha = 2*σ/sqrt(n_s-1)
    s = alpha*s;

    # obtain Markov transition matrix Π and its stationary distribution
    Π = rouwenhorst_Π(n_s,p);
    π = stationary_markov(Π);

    # s is log income, get income y and scale so that mean is 1
    y = exp.(s);
    y /= (π'*y);

    return(y,π,Π)
end

function backward_iteration(Va, Π, a_grid, y, r, β, eis)
    # step 1: discounting and expectations
    Wa = β*Π*Va;

    # step 2: solving for asset policy using the first-order condition
    c_endog = Wa.^(-eis);
    coh = repeat(y,outer=(1,length(a_grid))) + (1+r)*repeat(a_grid', outer=(length(y),1));

    a = Matrix{Float64}(undef,size(coh,1),size(coh,2));
    for s in 1:length(y)
        itp = LinearInterpolation(c_endog[s,:]+a_grid, a_grid, extrapolation_bc=Line());
        a[s,:] = itp.(coh[s,:]);
    end

    # step 3: enforcing the borrowing constraint and backing out consumption
    a = max.(a, a_grid[1]);
    c = coh - a;

    # step 4: using the envelope condition to recover the derivative of the value function
    Va = (1+r)*c.^(-1/eis)

    return(Va,a,c)
end

function policy_ss(Π, a_grid, y, r, β, eis, tol=1e-9)
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = repeat(y,outer=(1,length(a_grid))) + (1+r)*repeat(a_grid', outer=(length(y),1));
    c = 0.05*coh;
    Va = (1+r)*c.^(-1/eis);
    
    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    a = 0;
    a_old = 0;
    for i in 1:10_000
        (Va, a, c) = backward_iteration(Va, Π, a_grid, y, r, β, eis);

        # after iteration 0, can compare new policy function to old one
        if maximum(abs.(a .- a_old)) < tol
            break
        else
            a_old = a;
        end
    end
    return(Va, a, c)
end

function get_lottery(a, a_grid)
    # step 1: find the i such that a' lies between gridpoints a_i and a_(i+1)
    # note that the lower bound is on the true grid and endogenous grid
    # must be careful and assign all probability to it
    a_i = searchsortedfirst.(Ref(a_grid), a) .- 1;

    # step 2: obtain lottery probabilities π
    a_π = Matrix{Float64}(undef,size(a,1),size(a,2));
    for i in 1:size(a,1)
        for j in 1:size(a,2)
            if a_i[i,j] == 0;
                a_π[i,j] = 1.0;
            else
                a_π[i,j] = (a_grid[a_i[i,j]+1] - a[i,j])/(a_grid[a_i[i,j]+1] - a_grid[a_i[i,j]]);
            end
        end
    end

    return(a_i, a_π)
end

function forward_policy(D, a_i, a_π)
    a_i = max.(1,a_i);
    Dend = zeros(size(a_i,1),size(a_i,2));
    for s in 1:size(a_i,1)
        for a in 1:size(a_i,2)
            # send π(s,a) of the mass to gridpoint i(s,a)
            Dend[s,a_i[s,a]] += a_π[s,a]*D[s,a]

            # send 1-π(s,a) of the mass to gridpoint i(s,a)+1
            Dend[s,a_i[s,a]+1] += (1-a_π[s,a])*D[s,a]
        end
    end

    return(Dend)
end

function forward_iteration(D, Π, a_i, a_π)
    Dend = forward_policy(D, a_i, a_π);
    return(Π'*Dend)
end

function distribution_ss(Π, a, a_grid, tol=1e-10)
    (a_i, a_π) = get_lottery(a, a_grid);

    # as initial D, use stationary distribution for s, plus uniform over a
    π = stationary_markov(Π);
    D = repeat(π,outer=(1,length(a_grid)))./sum(repeat(π,outer=(1,length(a_grid))));

    # now iterate until convergence to acceptable threshold
    D_new = 0;
    for i in 1:10_000
        D_new = forward_iteration(D, Π, a_i, a_π)
        if maximum(abs.(D_new - D)) < tol
            break
        else
            D = D_new
        end
    end
    return(D_new)
end

function disequilibrium(r, a_min, a_max, a_size, ρ, σ, y_size, β, eis, α, δ)
    a_grid = discretize_assets(a_min, a_max, a_size);
    (y, π, Π) = discretize_income(ρ,σ,y_size);
    (Va, a, c) = policy_ss(Π, a_grid, y, r, β, eis, 1e-9)
    (a_i, a_π) = get_lottery(a, a_grid);
    D = distribution_ss(Π, a, a_grid);
    A = sum(D.*a);
    K = (α/(r+δ))^(1/(1-α));
    disequ = (K-A)^2;
    return(disequ)
end

function r_equ(a_min=0, a_max=10_000, a_size=500, ρ=0.975, σ=0.7, y_size=7, β=0.98, eis=1.0, α=0.36, δ=0.08)
    res = optimize(r -> disequilibrium(r, a_min, a_max, a_size, ρ, σ, y_size, β, eis, α, δ), -0.9999*δ, 0.9999*(1/β - 1), rel_tol = 1e-8);
    r_equ = Optim.minimizer(res);
    return(r_equ)
end

## Now, let's use specific parameters

a_min = 0;
#a_max = 1000;
a_max=347.54080382909143
#a_size = 500;
a_size = 150;
ρ = 0.975;
σ = 0.7;
y_size = 7;
β = 0.98;
eis = 1;
α = 0.36;
δ = 0.08;


function ayagari()
    r= r_equ(a_min, a_max, a_size, ρ, σ, y_size, β, eis, α, δ);
    a_grid = discretize_assets(a_min, a_max, a_size);
    (y, π, Π) = discretize_income(ρ,σ,y_size);
    (Va, a, c) = policy_ss(Π, a_grid, y, r, β, eis, 1e-9)
    D = distribution_ss(Π, a, a_grid);
    A = sum(D.*a);
    pdf=transpose(sum(D, dims=1))

    return r,D,A,pdf,a_grid

end

@time begin
(r,D,A,pdf,a_grid)=ayagari();
end

plot(a_grid[2:140],pdf[2:140],
linewidth = 3,
color=:red,
legend=false,
title="Wealth Distribution",
dpi=300
)
#savefig("wealth.png")