# SV model
# y_t = mu + sigma_v * exp(x_t / 2) * v_t
# x_t = phi * x_{t-1} + sigma_w * w_t
# where v_t and w_t have N(0,1) distribution.
# param = (mu, sigma_v, phi, sigma_w)

param.test = c(0.05, 0.05, 0.85, 0.35)
param.inf = c(-Inf, 1e-6, -0.999, 1e-6)
param.sup = c(Inf, Inf, 0.999, Inf)

generate.init = function(N, param) {
  # Generate initial state.
  init = param[4] / sqrt(1 - param[3]**2) * rnorm(N, 0, 1)
  return(init)
}

init.log.pdf = function(x, param) {
  # Pdf of initial state.
  m = 0
  s2 = param[4]**2 / (1 - param[3]**2)
  return(-0.5 * log(s2) - 0.5 * (x - m)**2 / s2)
}

deriv.init.log.pdf = function(x, param) {
  # Derivative of initial state pdf.
  m = 0
  s2 = param[4]**2 / (1 - param[3]**2)
  deriv.s2 = c(0, 0, 2 * param[3] * param[4]**2 / (1 - param[3]**2)**2, 2 * param[4] / (1 - param[3]**2))
  return(-0.5 * deriv.s2 / s2                   # -0.5 * log(s2)
         + 0.5 * deriv.s2 * (x - m)**2 / s2**2  # -0.5 * (x - m)**2 / s2)
  )
}

deriv2.init.log.pdf = function(x, param) {
  # 2nd derivative of initial state pdf.
  m = 0
  s2 = (param[4] / sqrt(1 - param[3]**2))**2
  deriv.s2 = matrix(c(0, 0, 2 * param[4]**2 * param[3] / (1 - param[3]**2)**4, 2 * param[4] / (1 - param[3]**2)), nrow=1)
  deriv2.s2 = matrix(c(0, 0, 0, 0,
                       0, 0, 0, 0,
		          			   0, 0, 2 * param[4]**2/ (1 - param[3]**2)**2 + 8 * param[3]**2 * param[4]**2 / (1 - param[3]**2)**3, 4 * param[3] * param[4] / (1 - param[3]**2)**2,
					             0, 0, 4 * param[3] * param[4] / (1 - param[3]**2)**2, 2 / (1 - param[3]**2)), nrow=4)
  return(-0.5 * deriv2.s2 / s2 + 0.5 * t(deriv.s2) %*% deriv.s2 / s2**2                           # -0.5 * deriv.s2 / s2
         + 0.5 * deriv2.s2 * (x - m)**2 / s2**2 - t(deriv.s2) %*% deriv.s2 * (x - m)**2 / s2**4)  # 0.5 * deriv.s2 * (x - m)**2 / s2**2
}

kernel.log.pdf = function(xprev, x, yprev, t.index, param) {
  # Pdf of state Markov kernel.
  one.row = matrix(1, 1, length(x))
  m = param[3] * xprev
  s2 = param[4]**2
  return(-0.5 * log(s2) - 0.5 * (t(one.row) %*% x - m %*% one.row)**2 / s2)
}

deriv.kernel.log.pdf = function(xprev, x, yprev, t.index, param) {
  # Derivative of state Markov kernel pdf.
  m = param[3] * xprev
  s2 = param[4]**2
  deriv.m = c(0, 0, xprev, 0)
  deriv.s2 = c(0, 0, 0, 2 * param[4])
  return(-0.5 * deriv.s2 / s2                   # -0.5 * log(s2)
         + 0.5 * deriv.s2 * (x - m)**2 / s2**2  # -0.5 * (t(one.row) %*% x - m %*% one.row)**2 / s2 (w.r.t. s2)
         + deriv.m * (x - m) / s2)              # -0.5 * (t(one.row) %*% x - m %*% one.row)**2 / s2 (w.r.t. m)
}

deriv2.kernel.log.pdf = function(xprev, x, yprev, t.index, param) {
  # 2nd derivative of state Markov kernel pdf.
  m = param[3] * xprev
  s2 = param[4]**2
  deriv.m = matrix(c(0, 0, xprev, 0), nrow=1)
  deriv.s2 = matrix(c(0, 0, 0, 2 * param[4]), nrow=1)
  deriv2.m = matrix(0, nrow=4, ncol=4)
  deriv2.s2 = matrix(c(0, 0, 0, 0,
                       0, 0, 0, 0,
          					   0, 0, 0, 0,
					             0, 0, 0, 2), nrow=4)
  return(-0.5 * deriv2.s2 / s2 + 0.5 * t(deriv.s2) %*% deriv.s2 / s2**2                          # -0.5 * deriv.s2 / s2
         + 0.5 * deriv2.s2 * (x - m)**2 / s2**2 - t(deriv.s2) %*% deriv.s2 * (x - m)**2 / s2**4  # 0.5 * deriv.s2 * (x - m)**2 / s2**2 (w.r.t. s2)
         - t(deriv.m) %*% deriv.s2 * (x - m) / s2**2                                             # 0.5 * deriv.s2 * (x - m)**2 / s2**2 (w.r.t. m)
         - t(deriv.s2) %*% deriv.m * (x - m) / s2**2                                             # deriv.m * (x - m) / s2) (w.r.t. s2)
         + deriv2.m * (x - m) / s2 - t(deriv.m) %*% deriv.m / s2)                                # deriv.m * (x - m) / s2)
}

generate.kernel = function(xprev, yprev, t.index, param) {
  # Generate state Markov process.
  N = length(xprev)
  x = param[3] * xprev + param[4] * rnorm(N, 0, 1)
  return(x)
}

observation.log.pdf = function(y, x, t.index, param) {
  # Pdf of observation distribution.
  m = param[1]
  s2 = param[2]**2 * exp(x)
  return(-0.5 * log(s2) - 0.5 * (y - m)**2 / s2)
}

deriv.observation.log.pdf = function(y, x, t.index, param) {
  # Derivative of observation distribution pdf.
  m = param[1]
  s2 = param[2]**2 * exp(x)
  deriv.m = c(1, 0, 0, 0)
  deriv.s2 = c(0, 2 * param[2] * exp(x), 0, 0)
  return(-0.5 * deriv.s2 / s2                   # -0.5 * log(s2)
         + 0.5 * deriv.s2 * (y - m)**2 / s2**2  # -0.5 * (y - m)**2 / s2 (w.r.t. s2)
         + deriv.m * (y - m) / s2)              # -0.5 * (y - m)**2 / s2 (w.r.t. m)
}

deriv2.observation.log.pdf = function(y, x, t.index, param) {
  # Derivative of observation distribution pdf.
  m = param[1]
  s2 = param[2]**2 * exp(x)
  deriv.m = matrix(c(1, 0, 0, 0), nrow=1)
  deriv.s2 = matrix(c(0, 2 * param[2] * exp(x), 0, 0), nrow=1)
  deriv2.m = matrix(0, nrow=4, ncol=4)
  deriv2.s2 = matrix(c(0, 0, 0, 0,
                       0, 2 * exp(x), 0, 0,
		          			   0, 0, 0, 0,
					             0, 0, 0, 0), nrow=4)
  return(-0.5 * deriv2.s2 / s2 + 0.5 * t(deriv.s2) %*% deriv.s2 / s2**2                          # -0.5 * deriv.s2 / s2
         + 0.5 * deriv2.s2 * (y - m)**2 / s2**2 - t(deriv.s2) %*% deriv.s2 * (y - m)**2 / s2**4  # 0.5 * deriv.s2 * (y - m)**2 / s2**2 (w.r.t. s2) 
         - t(deriv.s2) %*% deriv.m * (y - m) / s2**2                                             # 0.5 * deriv.s2 * (y - m)**2 / s2**2 (w.r.t. m) 
         - t(deriv.s2) %*% deriv.m * (y - m) / s2**2                                             # deriv.m * (y - m) / s2 (w.r.t. s2)
         + deriv2.m * (y - m) / s2 - t(deriv.m) %*% deriv.m / s2)                                # deriv.m * (y - m) / s2 (w.r.t. m)
}

generate.observation = function(N, x, t.index, param) {
  y = param[1] + param[2] * exp(x / 2) * rnorm(N, 0, 1)
  return(y)
}

initialize.parameters = function(y) {
  return(c(mean(y), sd(y), 0, 1))
}
