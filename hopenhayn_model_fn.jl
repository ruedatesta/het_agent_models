using Random, NLsolve, Plots, StatsPlots
using Optim, LinearAlgebra, LaTeXStrings
include("c:\\Users\\rhruedar\\Dropbox\\PhD\\het_agents\\rouwenhorst.jl")

## Parameters
# Example based on Chris Edmond's lectures

ρ=0.9;
logabar=1.39;
nstates=101;
mu=logabar
#mu=0;
σ=0.2;
α=2/3;
β=0.8;
k=20;
ke=40;
Dbar=100;

(agrid,trans,dist)=rouwenhorst(nstates,mu,σ,ρ);
agrid=exp.(agrid)

# Profit function
function profits(p,a)
    (1-α)*α^(α/(1-α))*(p*a).^(1/(1-α)).-k
end

# Production function
function production(a,p,α)
    a.*(p*a.*α).^(α/(1-α))
end

# Employment function
function employment(p,a,α)
    (p*a.*α).^(α/(1-α))
end



function incumbent(pguess,nstates,mu,σ,ρ)
    (agrid,trans,dist)=rouwenhorst(nstates,mu,σ,ρ);
    agrid=exp.(agrid)

    # INITIALIZE VALUE FUNCTION
    V=ones(nstates,1000)
    exits = Vector{Bool}(undef, nstates)
    Iter_Stop = 0;
    for j = 2:1000
        for i = 1:nstates
            #V[i,j] = profits(pguess,agrid[i])+ β* max.(0, (dist)'*V[:,j-1])
            V[i,j] = profits(pguess,agrid[i])+ β* max.(0, (trans[i,:])'*V[:,j-1])
        end
        if maximum(V[:,j] - V[:,j-1]) < 1e-6
            Iter_Stop = j-1 ;
            #for i = 1:nstates
            #~ ,ind = findmax(profits(pguess,agrid[i])+ max.(0, maximum(β*V[:,j-1])))
            #S[i] = K[ind]bn 
            #end
            exits=V[:,Iter_Stop].<0
            break
        end
    end
    vfi=V[:,Iter_Stop]

    return vfi,Iter_Stop,agrid,dist,trans,exits
    
end

function disequilibrium(pguess,nstates,mu,σ,ρ,ke)
    (vfi,Iter_Stop,agrid,dist,trans,exits)=incumbent(pguess,nstates,mu,σ,ρ)
    obj=(β*(dist)'*vfi-ke)^2
end

function free_entry(nstates,mu,σ,ρ,ke)
    
    res = optimize(pguess -> disequilibrium(pguess,nstates,mu,σ,ρ,ke), 0.00001, 10, rel_tol = 1e-8);
    pstar = Optim.minimizer(res);
    return pstar
end

@time begin
(pstar)=free_entry(nstates,mu,σ,ρ,ke)

(vfi,Iter_Stop,agrid,dist,trans,exits)=incumbent(pstar,nstates,mu,σ,ρ)

β*(dist)'*vfi

## Step 3. Measure of entrants

M=zeros(nstates,nstates);
# Typical element is (1-x_j (p*)) f_{ji}, where x is bool exits

for i=1:nstates
    for j=1:nstates
        M[i,j]=(1-exits[j])*trans[j,i];
    end
end

μ1=((I-M)^(-1))*dist; # mass of firms evaluated with m=1

D=Dbar/pstar

y=production(agrid,pstar,α);

m=D/(y'*μ1);

μ=m*((I-M)^(-1))*dist # Stationary mass of firms
end
## The model is solved. 
# Main statistics

Y=y'*μ # Agregate output
μs=μ/sum(μ) # Share of firms (they add to one)

e_aux=employment(pstar,agrid,α)
e=e_aux.*μ # Employment by productivity firm
es=e/sum(e) # Share employment (they add to one)

sum(agrid.*μ)

((exits')*μ)/sum(μ)


mass_continue=sum(M*μ) # mass of non exiters
mas_entry=sum(m.*dist) # mass of entrants

total_mass=mass_continue+mas_entry
mas_entry/sum(μ)

profits_aux=profits(pstar,agrid)
agg_profits=sum(profits_aux.*μ)

a_th_grid=searchsortedfirst.(Ref(vfi), 0)-1
a_th=agrid[a_th_grid]

entry_exit_rate=m/sum(μ)


# Identify firm with value function equal to 100 
maxplot1=searchsortedfirst.(Ref(vfi), 100);

# Value function and productivity
p1=plot(agrid[1:maxplot1],vfi[1:maxplot1],
linecolor=:blue,linewidth=2, legend=false,ylabel=L"\textrm{Value~function}~v(a_i,p^*)",
xlabel=L"\textrm{productivity} ~a_i")
annotate!(p1, 5.5, 100, text(L"v(a,p*)", :blue, :right, 10))
annotate!(p1, 6, -16, text(L"-k", :red, :right, 10))
annotate!(p1, 3.75,75, text(L"\textrm{cutoff}~ a(p*)", :black, :right, 10))
hline!(p1,[0 -k],linecolor=[:gray :red],linestyle=:dash)
vline!(p1,[a_th],linecolor=[:gray :red],linestyle=:dash)
plot!(p1,dpi=300)
savefig("figures\\value_fn_hopenhayn.png")

# Distribution of firms

maxplot2=searchsortedfirst.(Ref(agrid), 30)
p2=plot(agrid[1:maxplot2],μs[1:maxplot2],
linecolor=:blue,linewidth=2, legend=false,ylabel=L"\mathrm{prob~mass}",xlabel=L"\textrm{productivity} ~a_i")
plot!(p2,agrid[1:maxplot2],dist[1:maxplot2],linecolor=:red,linewidth=2)
plot!(p2,agrid[1:maxplot2],es[1:maxplot2],linecolor=:green,linewidth=2)
annotate!(p2, 12, 0.1, text("share firms", :blue, :right, 10))
annotate!(p2, 4, 0.085, text(L"\bar{f}_i", :red, :right, 10))
annotate!(p2, 13, 0.05, text("share employment", :green, :left, 10))
plot!(p2,dpi=300)
savefig("figures\\size_dist_hopenhayn.png")


#= What I learn
An increase in D does not affect the price. 
Does increase real outpu Y.

An increase in ke decrease entrants m, decreases entry/exit rate
=#