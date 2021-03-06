### scripts that relate to the moment updates.
# mit script fonts indicate variables that have a hat in the paper.



# all matrices are assumed to be dense matrices.
# term1 is inv_P0*m0.
function updatemoments( b::GaussianFlowSimpleBuffersType{T},
                        inv_P0::Matrix{T},
                        inv_P0_mul_m0::Vector{T},
                        inv_R::Matrix{T},
                        位::T)::Tuple{Vector{T},Matrix{T}} where T <: Real

    # set up.
    饾惢 = b.饾惢
    饾懄 = b.饾懄

    # update moments.
    饾憙 = inv(inv_P0 + 位*饾惢'*inv_R*饾惢)
    饾憙 = Utilities.forcesymmetric(饾憙)

    饾憵 = 饾憙*(inv_P0_mul_m0 + 位*饾惢'*inv_R*饾懄)

    return 饾憵, 饾憙
end

function updatemoments( b::GaussianFlowBlockDiagonalBuffersType{T},
                        inv_P0::Matrix{T},
                        inv_P0_mul_m0::Vector{T},
                        inv_R::Vector{Matrix{T}},
                        位::T)::Tuple{Vector{T},Matrix{T}} where T <: Real

    # set up.
    饾惢 = b.饾惢
    饾懄 = b.饾懄

    # 饾憙 = inv(inv_P0 + 位*饾惢'*inv_R*饾惢)
    A = applyrectmatrixcongruence(b.饾惢, inv_R)
    饾憙 = inv(inv_P0 + 位*A)
    饾憙 = Utilities.forcesymmetric(饾憙)

    #饾憵 = 饾憙*(inv_P0_mul_m0 + 位*饾惢'*inv_R*饾懄)
    c = applyHtRy(b.饾惢, inv_R, b.饾懄)
    饾憵 = 饾憙*(inv_P0_mul_m0 + 位*c)

    return 饾憵, 饾憙
end

function computelinearization!( b::GaussianFlowSimpleBuffersType{T},
                                鈭傁?::Function,
                                蠄::Function,
                                x::Vector{T},
                                y::Vector{T}) where T <: Real
    #
    b.饾惢[:] = 鈭傁?(x)
    b.蠄_eval[:] = 蠄(x)

    # 饾懄 = y - 蠄(x) + 鈭傁?(x)*x
    b.饾懄[:] = y - b.蠄_eval + b.饾惢*x

    return nothing
end

"""

y[n][m], n鈭圼N], m鈭圼M].
饾懄[m][n], n鈭圼N], m鈭圼M].

"""
function computelinearization!( b::GaussianFlowBlockDiagonalBuffersType{T},
                                鈭傁?::Function,
                                蠄::Function,
                                x::Vector{T},
                                y_set::Vector{Vector{T}}) where T <: Real


    #
    鈭傁坃鈭偽?::Vector{Vector{Vector{T}}} = 鈭傁?(x)
    b.鈭傁坃eval[:] = 鈭傁坃鈭偽?

    饾惢_set = 鈭傁坱oblockdiagmatrix(鈭傁坃鈭偽?) # I am here. change this to vector of matrices.
    b.饾惢_set[:] = 饾惢_set

    term3 = evalmatrixvectormultiply(饾惢_set, x)

    蠄_x::Vector{Vector{T}} = 蠄(x)
    b.蠄_eval[:] = 蠄_x

    # 饾懄 = y - 蠄(x) + 饾惢*x
    饾懄_set = collect( y_set[m] - 蠄_x[m] + term3[m] for m = 1:M )
    b.饾懄_set[:] = 饾懄_set

    return nothing
end
