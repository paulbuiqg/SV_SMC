# Modified SV-M model
# y_t = mu + lambda * x_t + sigma_v * exp(x_t / 2) * v_t
# x_t = phi * x_{t-1} + sigma_w * w_t
# where v_t and w_t ~ N(0,1).
# param = (mu, lambda, sigma_v, phi, sigma_w)

param = c(.05, 0.1, 0.65, 0.85, 0.35)
param.init = c(0, 0, 1, 0, 1)
param.inf = c(-Inf, -Inf, 1e-6, -0.999, 1e-6)
param.sup = c(Inf, Inf, Inf, 0.999, Inf)

generate.init = function(N, param=NULL) {
  # Generate initial state.
  init = param[5] / sqrt(1 - param[4]**2) * rnorm(N, 0, 1)
  return(init)
}

init.log.pdf = function(x, param=NULL) {
  # Pdf of initial state.
  m = 0
  s2 = (param[5] / sqrt(1 - param[4]**2))**2
  return(-0.5 * log(s2) - 0.5 * (x - m)**2 / s2)
}

kernel.log.pdf = function(xprev, x, yprev=NULL, t.index=NULL, param=NULL) {
  # Pdf of state Markov kernel.
  one.row = matrix(1, 1, length(x))
  m = param[4] * xprev
  s2 = param[5]**2
  return(-0.5 * log(s2) - 0.5 * (t(one.row) %*% x - m %*% one.row)**2 / s2)
}

generate.kernel = function(xprev, yprev=NULL, t.index=NULL, param=NULL) {
  # Generate state Markov process.
  N = length(xprev)
  x = param[4] * xprev + param[5] * rnorm(N, 0, 1)
  return(x)
}

observation.log.pdf = function(y, x, t.index=NULL, param=NULL) {
  # Pdf of observation distribution.
  m = param[1] + param[2] * x
  s2 = param[3]**2 * exp(x)
  return(-0.5 * log(s2) - 0.5 * (y - m)**2 / s2)
}

generate.observation = function(N, x, t.index=NULL, param=NULL) {
  y = param[1] + param[2] * x + param[3] * exp(x / 2) * rnorm(N, 0, 1)
  return(y)
}

initialize.parameters = function(y) {
  return(c(mean(y), 0, sd(y), 0, 1))
}