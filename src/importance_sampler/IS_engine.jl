
function unpackpmap(sol::Array{Tuple{ Array{Array{T,1},1},
                                      Array{Array{T,1},1},
                                      Array{Array{T,1},1}},1},
                    M::Int)::Tuple{Vector{Vector{T}},Vector{Vector{T}},Vector{Vector{T}}} where T <: Real

    N_batches = length(sol)

    x_array = Vector{Vector{T}}(undef,M)
    xp_array = Vector{Vector{T}}(undef,M)
    ln_wp_array =  Vector{Vector{T}}(undef,M)

    st::Int = 0
    fin::Int = 0
    for j = 1:N_batches

        st = fin + 1
        fin = st + length(sol[j][1]) - 1

        xp_array[st:fin] = sol[j][1]
        ln_wp_array[st:fin] = sol[j][2]
        x_array[st:fin] = sol[j][3]
    end

    return xp_array, ln_wp_array, x_array
end

function setupGFquantities( γ::T,
                            m_0::Vector{T},
                            P_0::Matrix{T},
                            R::Matrix{T},
                            y::Vector{T},
                            ψ::Function,
                            ∂ψ::Function,
                            ∂2ψ::Function,
                            ln_prior_pdf_func::Function,
                            ln_likelihood_func::Function) where T
    #
    p = GaussianFlowSimpleParamsType(m_0 = m_0,
                        P_0 = P_0,
                        R = R,
                        y = y,

                        inv_R = inv(R),
                        inv_P0_mul_m0 = P_0\m_0,
                        inv_P0 = inv(P_0))
    #
    m = GaussianFlowMethodsType( ψ = ψ,
                        ∂ψ = ∂ψ,
                        ∂2ψ = ∂2ψ,
                        ln_prior_pdf_func = ln_prior_pdf_func,
                        ln_likelihood_func = ln_likelihood_func)

    #
    b = GaussianFlowSimpleBuffersType(length(y), length(m_0), y[1])

    config = GaussianFlowConfigType(γ)

    return p, m, b, config
end

# simulates a separate Bλ_array and λ_array for each particle.
# Computes faster.
function traverseSDEs(  drawxfunc::Function,
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

        # This will get better sample diversity.
        λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, length(x))

        𝑥, 𝑤 = traversesdepathapproxflow(   x,
                                         λ_array,
                                         Bλ_array,
                                         problem_params,
                                         problem_methods,
                                         GF_buffers,
                                         GF_config)
        xp_array[n] = 𝑥[end]
        ln_wp_array[n] = [ 𝑤[end] ]
        x_array[n] = x
        #println("n = ", n)
    end

    return xp_array, ln_wp_array, x_array
end
