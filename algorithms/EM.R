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

EM.algo <- function(y, param.init, param.inf, param.sup, N, Nth, maxiter, tol) {
  # EM algorithm embedding particle filtering & smoothing.
  # The function Q to maximize is approximated by particle methods.
  param <- param.init
  param.seq <- c(param)
  # for (k in 1:maxiter) {
  #   filter = particle.filter(N, Nth, y, param)
  #   smoother = particle.smoother(filter$weights, filter$particles, param)
  #   opti.res = optimr(param, eval.minus.Q,
  #                     y=y,
  #                     wsmooth=smoother$weights,
  #                     w2smooth=smoother$weights2,
  #                     part=filter$particles,
  #                     method="L-BFGS-B", lower=param.inf, upper=param.sup)
  #   param = opti.res$par
  #   param.seq = cbind(param.seq, param)
  #   print(sprintf('- EM algorithm | iteration %i', k))
  # }
  # res.list = list("param.seq"=param.seq, "param"=param,
  #                 "particles"=filter$particles, "weights"=filter$weights)
  iter <- 0
  step.fail <- FALSE
  while (iter < maxiter && !step.fail) {
    iter <- iter + 1
    tryCatch(
      {
        filter <- particle.filter(N, Nth, y, param);
        smoother <- particle.smoother(filter$weights, filter$particles, param);
        opti.res <- optimr(param, eval.minus.Q,
                           y=y,
                           wsmooth=smoother$weights,
                           w2smooth=smoother$weights2,
                           part=filter$particles,
                           method="L-BFGS-B", lower=param.inf, upper=param.sup)
      },
      error = function(e) {step.fail <<- TRUE; print('EM algorithm | error')}
    )
    print(sprintf('- EM algorithm | iteration %i', iter))
  }
  res.list <- list("param.seq"=param.seq, "param"=param)
  return(res.list)
}