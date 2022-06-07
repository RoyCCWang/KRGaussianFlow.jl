
@everywhere include("../src/KRGaussianFlow.jl")
import .KRGaussianFlow

include("helpers.jl")

PyPlot.close("all")
fig_num = 1


# sample covmat:
xp_array_weighted = xp_array.*w_array
m_s = sum(xp_array_weighted)
Q = getcovmatfromparticles(xp_array, m_s, w_array)


# visualize.
if D_x == 2
    n_bins = 500
    fig_num = plot2Dhistogram(fig_num,
                                xp_array,
                                n_bins,
                                limit_a,
                                limit_b;
                                use_bounds = true,
                                title_string = "xp locations",
                                colour_code = "jet",
                                use_color_bar = true,
                                axis_equal_flag = true,
                                flip_vertical_flag = false)

end

println("x, generating = ", x_generating)
println()

xp1 = collect( xp_array[n][1] for n = 1:length(xp_array) )
xp2 = collect( xp_array[n][2] for n = 1:length(xp_array) )

##### test functional. mean.
A = [   0.85438   0.906057;
        0.906057  1.12264 ]
#f = xx->sinc(dot(xx,A*xx))^2
f = xx->xx[1]
f_eval_GF = evalexpectation(f, xp_array, w_array)

println("GF: 𝔼[f] over posterior   = ", f_eval_GF)
 # should be around 1.34 for 2D to 2D.
# NI answers:
# exampleψfunc2Dto2D1: 1.3699324768469774


println("NI test:")
@time f_eval_NI, val_h, err_h, val_Z, err_Z = evalexpectation(f,
                                 likelihood_func,
                                 prior_func,
         limit_a, limit_b, max_integral_evals, initial_div)

println("NI: 𝔼[f] over posterior   = ", f_eval_NI)
println("val_h = ", val_h)
println()

# got:
# preparing particles.
# 187.390355 seconds (1.37 M allocations: 68.901 MiB, 0.01% gc time)
# estimated ESS of Gaussian Flow: 22.578857939372952
#
# x, generating = [1.22, -0.35]
#
# GF: 𝔼[f] over posterior   = 1.3354071091016657
# NI test:
#  51.455690 seconds (879.18 M allocations: 40.474 GiB, 8.08% gc time)
# NI: 𝔼[f] over posterior   = 1.3489336018075437
# val_h = 0.02186045790028874
