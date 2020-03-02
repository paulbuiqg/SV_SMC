rm(list=ls())

path = '/home/paul/code/SV_SMC/'
path.model = paste(path, 'models/', sep='')
path.algo = paste(path, 'algorithms/', sep='')

source(paste(path.model, 'SV.R', sep=''))
source(paste(path.algo, 'SMC.R', sep=''))

series = generate.series(3000, param.test)
y = series$observations[2]
x = series$states[2]
yprev = series$observations[1]
xprev = series$states[1]
t.index = 1

eps = .001

fun = function(param) {init.log.pdf(x, param)}
deriv.fun = function(param) {deriv.init.log.pdf(x, param)}
deriv2.fun = function(param) {deriv2.init.log.pdf(x, param)}

# fun = function(param) {kernel.log.pdf(xprev, x, yprev, t.index, param)}
# deriv.fun = function(param) {deriv.kernel.log.pdf(xprev, x, yprev, t.index, param)}
# deriv2.fun = function(param) {deriv2.kernel.log.pdf(xprev, x, yprev, t.index, param)}

# fun = function(param) {observation.log.pdf(y, x, t.index, param)}
# deriv.fun = function(param) {deriv.observation.log.pdf(y, x, t.index, param)}
# deriv2.fun = function(param) {deriv2.observation.log.pdf(y, x, t.index, param)}

approx.deriv = function(fun, param, eps) {
  res = numeric(length(param))
  veps = numeric(length(param))
  for (i in 1:length(res)) {
    veps[i] = eps
    res[i] = (fun(param + veps) - fun(param - veps)) / (2 * eps)
    veps = numeric(length(param))
  }
  return(res)
}

approx.deriv2 = function(fun, param, eps) {
  res = matrix(0, length(param), length(param))
  veps = numeric(length(param))
  veps_ = numeric(length(param))
  for (i in 1:length(param)) {
    for (j in 1:i) {
      if (i==j) {
        veps[i] = eps
        res[i, i] = (fun(param + veps) - 2 * fun(param) + fun(param - veps)) / eps**2
      } else {
        veps[i] = eps
        veps[j] = eps
        veps_[i] = eps
        veps_[j] = -eps
        res[i, j] = (fun(param + veps) - fun(param + veps_) - fun(param - veps_) + fun(param - veps)) / (4 * eps**2)
        res[j, i] = res[i, j]
      }
      veps = numeric(length(param))
      veps_ = numeric(length(param))
    }
  }
  return(res)
}

deriv.fun(param.test)
approx.deriv(fun, param.test, eps)
deriv2.fun(param.test)
approx.deriv2(fun, param.test, eps)

