# #######################################################
# Parameters
# #######################################################

points = 10
epsilon = Float64(10.0^-6)
alpha = 0.05
alphas = exp10.(range(log10(10.0^(-4)), stop=log10(0.8), length=points))
alphas_nnzs_and_losses = [0.05]
rho = 0.0001
rhos = exp10.(range(log10(10.0^(-5)), stop=log10(10.0^(-2.5)), length=points))
rhos_nnzs_and_losses = [10^(-4.5)]
