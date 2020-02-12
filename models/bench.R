param = NULL

generate.init = function(N, param) {
  # Generate initial state.
  init = rnorm(N, 0, sqrt(10))
  return(init)
}

init.log.pdf = function(x, param) {
  # Pdf of initial state.
  return(-0.5 * x**2 / 10)
}

kernel.log.pdf = function(xprev, x, yprev, t.index, param) {
  # Pdf of state Markov kernel.
  one.row = matrix(1, 1, length(x))
  m = xprev / 2 + 25 * xprev / (1 + xprev**2) + 8 * cos(1.2 * t.index)
  return(-0.5 * (t(one.row) %*% x - m %*% one.row)**2 / 10)
}

generate.kernel = function(xprev, yprev, t.index, param) {
  # Generate state Markov process.
  N = length(xprev)
  x = xprev / 2 + 25 * xprev / (1 + xprev**2) + 8 * cos(1.2 * t.index) +
    rnorm(N, 0, sqrt(10))
  return(x)
}

observation.log.pdf = function(y, x, t.index, param) {
  # Pdf of observation distribution.
  m = x**2 / 20
  return(-0.5 * (y - m)**2)
}

generate.observation = function(N, x, t.index, param) {
  y = x**2 / 20 + rnorm(N, 0, 1)
  return(y)
}