
##### MC and numerical integration for verification.

# importance sampling.
function evalexpectation(f::Function,
                                X::Vector{Vector{T}},
                                w::Vector{T})::T where T <: Real
    #
    return sum( w[n]*f(X[n]) for n = 1:length(X) )
end

# more accurate, uses log-space weights.
function evalexpectation2(f::Function,
                                x_array::Vector{Vector{T}},
                                ln_w_array::Vector{T}) where T <: Real

    N = length(x_array)
    ln_N = log(N)
    ln_W = StatsFuns.logsumexp(ln_w_array)

    ln_out_positive = zeros(T, N)
    ln_out_negative = zeros(T, N)

    kp = 0
    kn = 0
    for i = 1:N
        f_x = f(x_array[i])

        if f_x > zero(T)
            kp += 1
            ln_out_positive[kp] = ln_w_array[i] + log(f_x)
        else
            kn += 1
            ln_out_negative[kn] = ln_w_array[i] + log(abs(f_x))
        end
    end
    resize!(ln_out_positive, kp)
    resize!(ln_out_negative, kn)


    # division by the sum of weights.
    out_positive = exp(StatsFuns.logsumexp(ln_out_positive) - ln_W)
    out_negative = exp(StatsFuns.logsumexp(ln_out_negative) - ln_W)


    return out_positive - out_negative, ln_out_positive, ln_out_negative
end


function evalexpectation(f::Function,
                            p_y_given_x::Function,
                            p_x::Function,
                            limit_a::Vector{T},
                            limit_b::Vector{T},
                            max_integral_evals::Int,
                            initial_div::Int) where T <: Real

    # prepare posterior.
    p_tilde = xx->p_y_given_x(xx)*p_x(xx)

    # # better-condition p_tilde0
    # X = uniformsampling(limit_a, limit_b, N)
    # Y = p_tilde0.(X)
    # p_tilde = xx->p_tilde(xx)/maximum(Y)
    #p_tilde = p_tilde0

    return evalexpectation(f, p_tilde, limit_a, limit_b, max_integral_evals, initial_div)
end

function evalexpectation(f::Function,
                            p_tilde::Function,
                            limit_a::Vector{T},
                            limit_b::Vector{T},
                            max_integral_evals::Int,
                            initial_div::Int) where T <: Real

    # get normalizing constant.
    val_Z, err_Z = evalintegral(p_tilde, limit_a, limit_b, max_integral_evals, initial_div)

    # integrand.
    h = xx->p_tilde(xx)*f(xx)

    # integrate.
    val_h, err_h = evalintegral(h, limit_a, limit_b, max_integral_evals, initial_div)

    return val_h/val_Z, val_h, err_h, val_Z, err_Z
end

function evalintegral( f::Function,
                        limit_a::Vector{T},
                        limit_b::Vector{T},
                        max_integral_evals::Int,
                        initial_div::Int) where T <: Real
    #
    @assert length(limit_a) == length(limit_b)


    return val_Z, err_Z = HCubature.hcubature( f, limit_a, limit_b;
                                            norm = norm, rtol = sqrt(eps(T)),
                                            atol = 0,
                                            maxevals = max_integral_evals,
                                            initdiv = initial_div )
end

function evalintegral( f::Function,
                        limit_a::T,
                        limit_b::T;
                        max_integral_evals::Int = 10000,
                        initial_div::Int = 1) where T <: Real

    return val_Z, err_Z = HCubature.hquadrature( f, limit_a, limit_b;
                                            norm = norm, rtol = sqrt(eps(T)),
                                            atol = 0,
                                            maxevals = max_integral_evals,
                                            initdiv = initial_div )
end


##### probability densities.

function evallnMVNlikelihood( x,
                            y::Vector{T},
                            m_x,
                            S_x,
                            ψ::Function,
                            S_y)::T where T <: Real

    #
    #term1 = evallnMVN(x, m_x, S_x)
    term1 = 0.0
    term2 = evallnMVN(y, ψ(x), S_y)

    return term1 + term2
end


function evallnMVN(x, μ::Vector{T}, Σ::Matrix{T})::T where T <: Real
    D = length(x)

    r = x-μ
    term1 = -0.5*dot(r,Σ\r)
    term2 = -D/2*log(2*π) -0.5*logdet(Σ)

    return term1 + term2
end

# w_array is assumed to be normalized such that it sums to 1.
function getcovmatfromparticles(x_array::Vector{Vector{T}},
                                m::Vector{T},
                                w_array::Vector{T}) where T <: Real

    @assert length(w_array) == length(x_array)

    D = length(x_array[1])
    @assert length(m) == D

    C = zeros(T,D,D)
    for i = 1:D
        for j = 1:D

            # add the contribution of each particle.
            for n = 1:length(x_array)
                contribution = (x_array[n][i] - m[i]) *(x_array[n][j] - m[j]) *w_array[n]
                C[i,j] += contribution
            end

        end
    end

    return C
end


###### viualize.
function plot2Dhistogram(fig_num::Int,
                        X::Vector{Vector{T}},
                        n_bins::Int,
                        limit_a::Vector{T},
                        limit_b::Vector{T};
                        use_bounds::Bool = true,
                        title_string::String = "",
                        colour_code::String = "Greys",
                        use_color_bar::Bool = true,
                        axis_equal_flag::Bool = true,
                        flip_vertical_flag::Bool = false)::Int where T <: Real

    PyPlot.figure(fig_num)
    fig_num += 1

    N_viz = length(X)
    p1 = collect(X[n][2] for n = 1:N_viz)
    p2 = collect(X[n][1] for n = 1:N_viz)

    bounds = [[limit_a[2], limit_b[2]], [limit_a[1], limit_b[1]]]

    if use_bounds
        PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap=colour_code)
    else
        PyPlot.plt.hist2d(p1, p2, n_bins, cmap=colour_code)
    end

    if use_color_bar
        PyPlot.plt.colorbar()
    end

    if axis_equal_flag
        PyPlot.plt.axis("equal")
    end

    PyPlot.title(title_string)

    if flip_vertical_flag
        PyPlot.plt.gca().invert_yaxis()
    end

    return fig_num
end

##### others
# numerical derivatives.
function eval∂2ψND( x::Vector{T}, ψ::Function, D_y::Int)::Vector{Vector{T}} where T

    D_x = length(x)

    # get component functions
    ψ_components = Vector{Function}(undef, D_y)
    for j = 1:D_y
        ψ_components[j] = xx->ψ(xx)[j]
    end

    # get second derivatives.
    ∂2ψ_x = Vector{Vector{T}}(undef, D_y)

    for j = 1:D_y
        A = Calculus.hessian(ψ_components[j], x)
        ∂2ψ_x[j] = Utilities.packageuppertriangle(A)
    end


    return ∂2ψ_x
end
