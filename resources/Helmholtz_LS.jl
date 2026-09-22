using Clapeyron
import Clapeyron: @registermodel, AssocOptions, has_sites, getparams
import Clapeyron: N_A, k_B
import Clapeyron: sigma_LorentzBerthelot, epsilon_LorentzBerthelot, assoc_mix
import Clapeyron: Solvers

struct LSParam <: EoSParam
    N::SingleParam
    Npm::SingleParam
    Nnc::SingleParam
    Z::SingleParam
    Znet::SingleParam
    chi::Float64
end

abstract type LSModel <: EoSModel end

struct LS <: LSModel
    components::Vector{String}
    params::LSParam
end

LS
export LS

function LS(components, chi=0.;
    userlocations=String[])
    params = getparams(components, [""]; userlocations=userlocations)
    N = params["N"]
    Npm = params["Npm"]
    Nnc = params["Nnc"]
    Z = params["Z"]
    Znet = params["Znet"]
    packagedparams = LSParam(N,Npm,Nnc,Z,Znet,chi)
    model = LS(components, packagedparams)
    return model
end

# @registermodel LS

function a(model::LSModel,lB,ρ)
    N           = model.params.N.values
    Npm         = model.params.Npm.values
    Nnc         = model.params.Nnc.values
    Z           = model.params.Z.values
    Znet        = model.params.Znet.values
    ν           = @. abs(Znet/N)
    χ           = model.params.chi

    # lB          = 1/T/4π
        
    f0          = sum(@. ρ./N*(log(ρ./N)-1))
    η           = (π/6)*sum(ρ)
    fhs         = 6η^2*(4-3η)/(π*(1-η)^2)
    
    if lB == 0.0
        fel = 0.0

        yhs         = (2+η)/(2*(1-η)^2)
        fch         = sum(ρ./N.*((1 .- N).*log.(yhs)))
    else
        κ           = sqrt(4π*lB*sum(ν.*ρ.*Z.^2))
        Γ           = (-1+sqrt(1+2κ))/(2)
        fel         = -Γ^3*(2/3+Γ)/π
        # fel         = κ^3/12pi
        
        yhs         = (2+η)/(2*(1-η)^2)
        ypp         = yhs*exp.(-lB.*Z.^2/(1+Γ)^2+lB.*Z.^2)
        ypm         = yhs*exp.(lB.*Z.^2/(1+Γ)^2-lB.*Z.^2)
        # println("ypp = ", ypp)
        fch         = sum(ρ./N.*((1 .+ Npm .+ Nnc .- N).*log.(ypp) .- Npm.*log.(ypm) .- Nnc.*log.(yhs)))
    end

    # χ           = A+B/T
    if length(ρ) >= 3
        fchi        = χ*(ρ[1]+ρ[2])*(ρ[3])
    else
        fchi        = 0
    end

    return (f0+fhs+fel+fch+fchi)
end

function a_el(model::LSModel,lB,ρ)
    N           = model.params.N.values
    Npm         = model.params.Npm.values
    Nnc         = model.params.Nnc.values
    Z           = model.params.Z.values
    Znet        = model.params.Znet.values
    ν           = @. abs(Znet/N)
    η           = (π/6)*sum(ρ)

    κ           = sqrt(4π*lB*sum(ν.*ρ.*Z.^2))
    Γ           = (-1+sqrt(1+2κ))/(2)
    fel         = -Γ^3*(2/3+Γ)/π
    return fel
end

function a_ch(model::LSModel,lB,ρ)
    N           = model.params.N.values
    Npm         = model.params.Npm.values
    Nnc         = model.params.Nnc.values
    Z           = model.params.Z.values
    Znet        = model.params.Znet.values
    ν           = @. abs(Znet/N)
    η           = (π/6)*sum(ρ)

    κ           = sqrt(4π*lB*sum(ν.*ρ.*Z.^2))
    Γ           = (-1+sqrt(1+2κ))/(2)
    fel         = -Γ^3*(2/3+Γ)/π
    # fel         = κ^3/12pi
    
    yhs         = (2+η)/(2*(1-η)^2)
    ypp         = yhs*exp.(-lB.*Z.^2/(1+Γ)^2+lB.*Z.^2)
    ypm         = yhs*exp.(lB.*Z.^2/(1+Γ)^2-lB.*Z.^2)
    # println("ypp = ", ypp)
    fch         = sum(ρ./N.*((1 .+ Npm .+ Nnc .- N).*log.(ypp) .- Npm.*log.(ypm) .- Nnc.*log.(yhs)))
    return fch
end

function X(model::LSModel, lB, ρ)
    n           = model.params.n.values
    λ           = model.params.lambda
    
        
    A = n[1]*ρ[1]*λ
    B = n[2]*ρ[2]*λ
    Ap = n[3]*ρ[3]*λ

    X0 = 1.0*ones(eltype(first(ρ)+lB), 3)

    if n[1] == 1 && n[2] == 1 && n[3] == 0
        X₊ = (-(1+(Ap-A)) + sqrt((1+(Ap-A))^2+4A))/(2A)
        X₋ = 1/(1 + A*X₊)
        Xₚ = X0[3]
    elseif n[1] == 1 && n[2] == 0 && n[3] == 1
        X₊ = (-(1+(Ap-A)) + sqrt((1+(Ap-A))^2+4A))/(2A)
        Xₚ = 1/(1 + A*X₊)
        X₋ = X0[2]
    elseif n[1] == 1 || n[2] == 1
        g_pair!(Xnew, Xold) = Xsolv_pair!(Xnew, Xold, [A, B, Ap], n)
        (X₊, X₋, Xₚ) = Solvers.fixpoint(g_pair!,X0,Solvers.SSFixPoint(dampingfactor = 0.1); max_iters=10000)
    else
        g_full!(Xnew, Xold) = Xsolv_full!(Xnew, Xold, [A, B, Ap], n)
        (X₊, X₋, Xₚ) = Solvers.fixpoint(g_full!,X0,Solvers.SSFixPoint(dampingfactor = 0.1); max_iters=10000)
    end
    return (X₊, X₋, Xₚ)
end

function Xsolv_pair!(Xnew,Xold,K,n)
    n1 = n[1]
    n2 = n[2]

    A = K[1]*Xold[1]
    B = K[2]*Xold[2]
    Ap = K[3]*Xold[3]

    C = (1-A^(n2-1))/(1-A)
    D = (1-B^(n1-1))/(1-B)

    Xnew .= [1/(1 + Ap + B + A*B*C + B^2*D), 
             1/(1 + A + A*B*D + A^2*C),
             1/(1 + A)]
    return Xnew
end

function Xsolv_full!(Xnew,Xold,K,n)
    n1 = n[1]
    n2 = n[2]

    A = K[1]*Xold[1]
    B = K[2]*Xold[2]
    Ap = K[3]*Xold[3]

    C = (1-A^(n2-1))/(1-A)
    D = (1-B^(n1-1))/(1-B)
    E = (1-A^(2n2-2))/(1-A)*(1-B^(2n1-2))/(1-B)

    Xnew .= [1/(1 + Ap + B + A*B*C + B^2*D + 2*A*B^2*E), 
             1/(1 + A + A*B*D + A^2*C + 2*A^2*B*E),
             1/(1 + A)]
    return Xnew
end

function g(model,lB, ρ)
    N = model.params.N.values
    Z = model.params.Z.values
    ν = abs(Z[1]/Z[2])

    fun(x)  = a(model,lBc,[x[1],x[1]/N[1]*ν,1-x[1]-x[1]/N[1]*ν])
    df(x) = Clapeyron.ForwardDiff.derivative(fun,x)
    return df(ρ[1])*ρ[1]
end