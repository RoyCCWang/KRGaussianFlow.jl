# test exact Gaussian flow.

using Distributed

@everywhere import HCubature
@everywhere import Utilities

@everywhere import Printf
@everywhere import PyPlot
@everywhere import Random

@everywhere using LinearAlgebra
@everywhere import Interpolations

@everywhere using FFTW

@everywhere import Statistics

@everywhere import Distributions

@everywhere import Calculus

@everywhere import ForwardDiff

@everywhere import StatsFuns

@everywhere include("../src/KRGaussianFlow.jl")
import .KRGaussianFlow


@everywhere include("example_funcs.jl")
@everywhere include("helpers.jl")

#import VisualizationTools

#@everywhere import Seaborn
# @everywhere include("../src/misc/declarations.jl")
#
# @everywhere include("../src/SDE/Brownian.jl")
# @everywhere include("../src/flows/approx_flow.jl")
# @everywhere include("../tests/routines/sde.jl")
# @everywhere include("../src/flows/moments.jl")
# @everywhere include("../src/flows/derivatives.jl")
# @everywhere include("../src/misc/utilities.jl")
# @everywhere include("../src/misc/utilities2.jl")
#
# @everywhere include("../tests/routines/simulation_tools.jl")
# @everywhere include("../src/flows/exact_flow.jl")
#
# @everywhere include("../src/importance_sampler/IS_engine.jl")
# @everywhere include("../src/diagnostics/functionals.jl")


PyPlot.close("all")
fig_num = 1

Random.seed!(25)

N_batches = 16



#
max_integral_evals = typemax(Int) #1000000
initial_div = 1000

#demo_string = "mixture"
#demo_string = "normal"

# D_x = 2
# D_y = 3
# ψ = exampleψfunc2Dto3D1

D_x = 2
D_y = 2
ψ = exampleψfunc2Dto2D1

# D_x = 4
# D_y = 2
# ψ = exampleψfunc4Dto2D1

limit_a = ones(Float64, D_x) .* -10.9
limit_b = ones(Float64, D_x) .* 10.9

# oracle latent variable, x.
x_generating = [1.22; -0.35]

# observation model aside from ψ.

σ = 0.02
R = diagm( 0 => collect(σ for d = 1:D_y) )

# generate observation.
true_dist_y = Distributions.MvNormal(ψ(x_generating),R)
y = rand(true_dist_y)

# prior.
m_0 = randn(Float64, D_x)
P_0 = Utilities.generaterandomposdefmat(D_x)

prior_dist = Distributions.MvNormal(m_0, P_0)


prior_func = xx->exp(Distributions.logpdf(prior_dist, xx))

# function of x!
likelihood_func = xx->exp(evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R))


∂ψ = xx->Calculus.jacobian(ψ, xx, :central)
#∂ψ = xx->ForwardDiff.jacobian(ψ, xx)
D_y = length(y)
∂2ψ = xx->eval∂2ψND(xx, ψ, D_y)

ln_prior_pdf_func = xx->Distributions.logpdf(prior_dist, xx)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

### flow.

## set up SDE.
N_discretizations = 1000
#γ = 0.1/2
N_particles = 10000

# # set up Brownian motion.
#λ_array, Bλ_array = KRGaussianFlow.drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)

### traverse the SDE for each particle.
println("preparing particles.")
drawxfunc = xx->rand(prior_dist)
@time xp_array, ln_wp_array, x_array = KRGaussianFlow.paralleltraverseSDEs(drawxfunc,
                            N_discretizations,
                            #γ,
                            m_0,
                            P_0,
                            R,
                            y,
                            ψ,
                            ∂ψ,
                            ∂2ψ,
                            ln_prior_pdf_func,
                            ln_likelihood_func,
                            N_particles,
                            N_batches)
#

# normalize weights.
ln_W = StatsFuns.logsumexp(ln_wp_array)
w_array = collect( exp(ln_wp_array[n] - ln_W) for n = 1:N_particles )

ln_w_sq_array = collect( 2*ln_wp_array[n] - 2*ln_W for n = 1:N_particles )
ESS_GF = 1/exp(StatsFuns.logsumexp(ln_w_sq_array))

println("estimated ESS of Gaussian Flow: ", ESS_GF)
println()
