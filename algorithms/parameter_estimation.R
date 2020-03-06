library(optimr)

eval.Q = function(param, y, wsmooth, w2smooth, part) {
  # Particle approximation of function Q.
  # This function is to be maximized in the EM algorithm.
  T = length(part[,1])
  I1 = sum(wsmooth[1,] * init.log.pdf(part[1,], param))
  I2 = 0
  I3 = sum(wsmooth[1,] * observation.log.pdf(y[1], part[1,], 1, param))
  for (t in 2:T) {
    I2 = I2 + sum(w2smooth[t-1,,] * kernel.log.pdf(part[t-1,], part[t,], y[t-1], t, param))
    I3 = I3 + sum(wsmooth[t,] * observation.log.pdf(y[t], part[t,], t, param))
  }
  Q = I1 + I2 + I3
  return(Q)
}

eval.minus.Q = function(param, y, wsmooth, w2smooth, part) {
  # Negative version of above.
  minus.Q = - eval.Q(param, y, wsmooth, w2smooth, part)
  return(minus.Q)
}

EM.algo = function(y, param.init, param.inf, param.sup, N, Nth, maxiter, tol) {
  # EM algorithm embedding particle filtering & smoothing.
  # The function Q to maximize is approximated by particle methods.
  param = param.init
  param.seq = c(param)
  iter = 0
  while (iter < maxiter) {
    iter = iter + 1
    tryCatch(
      {
        filter = particle.filter(N, Nth, y, param);
        smoother = particle.smoother(filter$weights, filter$particles, y, param);
        opti.res = optimr(param, eval.minus.Q,
                          y=y,
                          wsmooth=smoother$weights,
                          w2smooth=smoother$weights2,
                          part=filter$particles,
                          method="L-BFGS-B", lower=param.inf, upper=param.sup)
      },
      error = function(e) {print('EM algorithm | error'); break}
    )
    if (all(is.finite(opti.res$par))) {
      param = opti.res$par
      param.seq = cbind(param.seq, param)
      print(sprintf('- EM algorithm | iteration %i', iter))
    } else {
      print('EM algorithm | error | NaN of Inf parameter')
      break
    }
  }
  return(list("param.seq"=param.seq, "param"=param))
}


gradient.descent.estimation = function(N, Nth, y, param.init, thres=.001, maxiter=100) {
  # Search zero of the likelihood gradient (score).
  T = length(y)
  param = param.init
  param.seq = c(param)
  iter = 1
  r = thres + 10
  while (r > thres & iter < maxiter) {
    iter = iter + 1
    score = score.particle.filter(N, Nth, y, param)$scores[T,]
    param = param - (1 / iter)**(2 / 3) * score
    param.seq = cbind(param.seq, param)
    r = sqrt(sum((param.seq[,iter] - param.seq[,iter - 1])**2) / sum(param.seq[,iter - 1]**2))
    print(sprintf('- Gradient descent | iteration %i', iter))
  }
  return(list("param.seq"=param.seq, "param"=param))
}
