
function computeflowparticleapproxstate(位_a::T,
                                        位_b::T,
                                        x::Vector{T},
                                        problem_params,
                                        problem_methods,
                                        GF_buffers,
                                        GF_config)::Vector{T} where T <: Real
    # set up
    饾憵_a = GF_buffers.饾憵_a
    饾憙_a = GF_buffers.饾憙_a
    饾憵_b = GF_buffers.饾憵_b
    饾憙_b = GF_buffers.饾憙_b

    term1 = 饾憵_b
    #C = 饾憙_b*inv(饾憙_a)
    C = Utilities.forcesymmetric(饾憙_b*inv(饾憙_a))

    term2 = real.(LinearAlgebra.sqrt(C))*(x - 饾憵_a)

    return term1 + term2
end

# 纬 set to 0.
function computelnabsdetJofstateupdate(  位_a::T,
                                        位_b::T,
                                        problem_params,
                                         problem_methods,
                                         GF_buffers::GaussianFlowSimpleBuffersType{T},
                                         GF_config,
                                        x::Vector{T}) where T <: Real
    # parse.
    饾惢 = GF_buffers.饾惢
    饾憙_a = GF_buffers.饾憙_a
    饾憙_b = GF_buffers.饾憙_b
    饾憵_a = GF_buffers.饾憵_a
    饾憵_b = GF_buffers.饾憵_b
    饾懄 = GF_buffers.饾懄

    R = problem_params.R
    纬 = GF_config.纬

    鈭傪潙歘a_鈭倄 = GF_buffers.鈭傪潙歘a_鈭倄
    鈭傪潙歘b_鈭倄 = GF_buffers.鈭傪潙歘b_鈭倄

    鈭傪潙僟b_鈭倄 = GF_buffers.鈭傪潙僟b_鈭倄
    鈭傪潙僟b_inv饾憙_a_鈭倄 = GF_buffers.鈭傪潙僟b_inv饾憙_a_鈭倄

    鈭傪潙僟b_sqrt_鈭倄 = GF_buffers.鈭傪潙僟b_sqrt_鈭倄
    鈭傪潙僟b_inv饾憙_a_sqrt_鈭倄 = GF_buffers.鈭傪潙僟b_inv饾憙_a_sqrt_鈭倄

    # set up.
    #鈭傪潗籺_鈭倄::Vector{Matrix{T}} = get鈭傪潗籺func(x_a)

    D_x = length(饾憵_a)
    螖位 = 位_b - 位_a
    #螖系 = 系_b - 系_a
    x_minus_饾憵_a = x - 饾憵_a
    #饾憙_b_inv饾憙_a = 饾憙_b*inv(饾憙_a)

    # other recurring factors.
    factor12 = real.(LinearAlgebra.sqrt(饾憙_b*inv(饾憙_a)))

    #exp_factor = sqrt( (one(T) - exp(-纬*螖位))/螖位 )

    # first term.
    J = 鈭傪潙歘b_鈭倄 + factor12*(LinearAlgebra.I - 鈭傪潙歘a_鈭倄)

    # the other terms.
    for i = 1:D_x
        for j = 1:D_x

            term3 = sum( 鈭傪潙僟b_inv饾憙_a_sqrt_鈭倄[j][i,k]*x_minus_饾憵_a[k] for k = 1:D_x )

            J[i,j] = J[i,j] + term3

        end
    end

    return logabsdet(J)[1]
end

function computeflowparticleapproxweight(位_a::T,
                                        位_b::T,
                                        x_a::Vector{T},
                                        x_b::Vector{T},
                                        problem_params,
                                        problem_methods,
                                        GF_buffers,
                                        GF_config,
                                        ln_w_a::T) where T <: Real
    #
    # set up
    纬 = GF_config.纬
    inv_P0_mul_m0 = problem_params.inv_P0_mul_m0
    inv_P0 = problem_params.inv_P0
    inv_R = problem_params.inv_R
    y = problem_params.y

    蠄 = problem_methods.蠄
    鈭傁? = problem_methods.鈭傁?
    鈭?2蠄 = problem_methods.鈭?2蠄
    ln_prior_pdf_func = problem_methods.ln_prior_pdf_func
    ln_likelihood_func = problem_methods.ln_likelihood_func

    #
    numerator1 = ln_prior_pdf_func(x_b)
    numerator2 = ln_likelihood_func(x_b)*位_b

    denominator1  = ln_prior_pdf_func(x_a)
    denominator2 = ln_likelihood_func(x_a)*位_a

    lnabsdetJ = computelnabsdetJofstateupdate( 位_a,
                                    位_b,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config,
                                    x_a)

    #
    # println("lnabsdetJ = ", lnabsdetJ)
    # @assert 1==2

    ln_w_b_unnorm = ln_w_a + numerator1 + numerator2 + lnabsdetJ - denominator1 - denominator2


    return ln_w_b_unnorm
end

# 纬 set to 0.
function traversesdepathapproxflow2( x0::Vector{T},
                                    位_array::LinRange{T},
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config) where T <: Real

    # set up.
    N_steps = length(位_array)
    #D = length(x0)
    #B_位, drawfunc = setupdrawBrownia(D, one(T))

    # drawfunc(螖位)

    ln_w0 = zero(T)

    # allocate.
    x = Vector{Vector{T}}(undef, N_steps)
    ln_w = Vector{T}(undef,N_steps)

    # to start.
    #系_a = zeros(T, length(B位_array[1]))
    #系_b = B位_array[1]
    位_a = zero(T)
    位_b = 位_array[1]

    # update.
    updateGFbuffers!(  problem_params,
                            problem_methods,
                            GF_buffers,
                            GF_config,
                            位_a,
                            位_b,
                            x0)

    x[1] = computeflowparticleapproxstate(  位_a,
                                            位_b,
                                            x0,
                                            problem_params,
                                            problem_methods,
                                            GF_buffers,
                                            GF_config)

    ln_w[1] = computeflowparticleapproxweight(  位_a,
                                                位_b,
                                                x0,
                                                x[1],
                                                problem_params,
                                                problem_methods,
                                                GF_buffers,
                                                GF_config,
                                                ln_w0)
    #
    # println("x[1] = ", x[1])
    # println("ln_w[1] = ", ln_w[1])
    # @assert 1==2

    for n = 2:length(x)
        位_a = 位_array[n-1]
        位_b = 位_array[n]

        #
        updateGFbuffers!(  problem_params,
                                problem_methods,
                                GF_buffers,
                                GF_config,
                                位_a,
                                位_b,
                                x[n-1])

        x[n] = computeflowparticleapproxstate(  位_a,
                                                位_b,
                                                x[n-1],
                                                problem_params,
                                                problem_methods,
                                                GF_buffers,
                                                GF_config)

        #
        ln_w[n] = computeflowparticleapproxweight(  位_a,
                                                    位_b,
                                                    x[n-1],
                                                    x[n],
                                                    problem_params,
                                                    problem_methods,
                                                    GF_buffers,
                                                    GF_config,
                                                    ln_w[n-1])
    end

    return x, ln_w
end

"""
    bar(x[, y])

Compute the Bar index between `x` and `y`. If `y` is missing, compute
the Bar index between all pairs of columns of `x`.

# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""
function paralleltraverseSDEs(  drawxfunc::Function,
                                N_discretizations::Int,
                                纬::T,
                                m_0::Vector{T},
                                P_0::Matrix{T},
                                R,
                                y,
                                蠄::Function,
                                鈭傁?::Function,
                                鈭?2蠄::Function,
                                ln_prior_pdf_func::Function,
                                ln_likelihood_func::Function,
                                M::Int,
                                N_batches::Int) where T <: Real

    # # work on intervals.
    # M_for_each_batch = Vector{Int}(undef, N_batches)
    # 饾憖::Int = round(Int, M/N_batches)
    # fill!(M_for_each_batch, 饾憖)
    # M_for_each_batch[end] = abs(M - (N_batches-1)*饾憖)
    #
    # @assert M == sum(M_for_each_batch) # sanity check.

    # takes 2.
    M_for_each_batch = Vector{Int}(undef, N_batches)

    fill!(M_for_each_batch, div(M,N_batches))

    N_batches_w_extra = mod(M,N_batches)
    for i = 1:N_batches_w_extra
        M_for_each_batch[i] += 1
    end
    @assert M == sum(M_for_each_batch) # sanity check.

    # set up neccessary objects.
    problem_params,
        problem_methods,
        GF_buffers,
        GF_config = setupGFquantities( 纬,
                                m_0,
                                P_0,
                                R,
                                y,
                                蠄,
                                鈭傁?,
                                鈭?2蠄,
                                ln_prior_pdf_func,
                                ln_likelihood_func)

    ##prepare worker function.
    workerfunc = xx->traverseSDEs(drawxfunc,
                                    N_discretizations,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config,
                                    xx)

    # compute solution.
    sol = pmap(workerfunc, M_for_each_batch )

    # unpack solution.
    xp_array, ln_wp_array, x_array = unpackpmap(sol, M)

    ln_wp_array = collect( ln_wp_array[n][1] for n = 1:length(ln_wp_array))

    return xp_array, ln_wp_array, x_array
end


# 纬 set to zero.
function paralleltraverseSDEs(  drawxfunc::Function,
                                N_discretizations::Int,
                                m_0::Vector{T},
                                P_0::Matrix{T},
                                R,
                                y,
                                蠄::Function,
                                鈭傁?::Function,
                                鈭?2蠄::Function,
                                ln_prior_pdf_func::Function,
                                ln_likelihood_func::Function,
                                M::Int,
                                N_batches::Int) where T <: Real

    # # work on intervals.
    # M_for_each_batch = Vector{Int}(undef, N_batches)
    # 饾憖::Int = round(Int, M/N_batches)
    # fill!(M_for_each_batch, 饾憖)
    # M_for_each_batch[end] = abs(M - (N_batches-1)*饾憖)
    #
    # @assert M == sum(M_for_each_batch) # sanity check.

    # takes 2.
    M_for_each_batch = Vector{Int}(undef, N_batches)

    fill!(M_for_each_batch, div(M,N_batches))

    N_batches_w_extra = mod(M,N_batches)
    for i = 1:N_batches_w_extra
        M_for_each_batch[i] += 1
    end
    @assert M == sum(M_for_each_batch) # sanity check.

    # set up neccessary objects.
    problem_params,
        problem_methods,
        GF_buffers,
        GF_config = setupGFquantities( zero(T),
                                m_0,
                                P_0,
                                R,
                                y,
                                蠄,
                                鈭傁?,
                                鈭?2蠄,
                                ln_prior_pdf_func,
                                ln_likelihood_func)
    #
    位_array = LinRange(1/(N_discretizations-1), one(T), N_discretizations)


    ##prepare worker function.
    workerfunc = xx->traverseSDEs2(位_array, drawxfunc,
                                    N_discretizations,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config,
                                    xx)

    # compute solution.
    sol = pmap(workerfunc, M_for_each_batch )

    # unpack solution.
    xp_array, ln_wp_array, x_array = unpackpmap(sol, M)

    ln_wp_array = collect( ln_wp_array[n][1] for n = 1:length(ln_wp_array))

    return xp_array, ln_wp_array, x_array
end

# simulates a separate B位_array and 位_array for each particle.
# Computes faster.
function traverseSDEs2(  位_array, drawxfunc::Function,
                        N_discretizations::Int,
                        problem_params::GaussianFlowSimpleParamsType{T},
                        problem_methods::GaussianFlowMethodsType,
                        GF_buffers::GaussianFlowSimpleBuffersType{T},
                        GF_config::GaussianFlowConfigType{T},
                        N_particles::Int)::Tuple{Vector{Vector{T}},
                                                Vector{Vector{T}},
                                                Vector{Vector{T}}} where T <: Real


    # allocate outputs.
    x_array = Vector{Vector{T}}(undef, N_particles)
    xp_array = Vector{Vector{T}}(undef, N_particles)
    ln_wp_array = Vector{Vector{T}}(undef, N_particles)

    # traverse particles.
    for n = 1:N_particles
        x = drawxfunc(1.0)

        饾懃, 饾懁 = traversesdepathapproxflow2(   x,
                                         位_array,
                                         problem_params,
                                         problem_methods,
                                         GF_buffers,
                                         GF_config)
        xp_array[n] = 饾懃[end]
        ln_wp_array[n] = [ 饾懁[end] ]
        x_array[n] = x
        #println("n = ", n)
    end

    return xp_array, ln_wp_array, x_array
end
