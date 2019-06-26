generate.series = function(T, param=NULL) {
  # Generate state & observation sequences.
  x = numeric(T)
  y = numeric(T)
  x[1] = generate.init(1, param)
  y[1] = generate.observation(1, x[1], 1, param)
  for (t in 2:T) {
    x[t] = generate.kernel(x[t-1], y[t-1], t, param)
    y[t] = generate.observation(1, x[t], t, param)
  }
  res.list = list("states" = x, "observations" = y)
  return(res.list)
}

observation.init = function(param=NULL) {
  # Generate initial state & observation.
  x = generate.init(1, param)
  y = generate.observation(1, x[1], 1, param)
  res.list = list("state" = x, "observation" = y)
  return(res.list)
}

observation.step = function(xprev, yprev=NULL, t.index=NULL, param=NULL) {
  # Generate next state & observation.
  x = generate.kernel(xprev, yprev, t.index, param)
  y = generate.observation(1, x, t.index, param)
  res.list = list("state" = x, "observation" = y)
  return(res.list)
}

particle.filter = function(N, Nth, y, param=NULL) {
  # Generic particle filter.
  T = length(y)
  part = matrix(0, T, N)
  w = matrix(0, T, N)
  part[1,] = generate.init(N, param)
  wei = exp(observation.log.pdf(y[1], part[1,], 1, param))
  w[1,] = wei / sum(wei)
  t = 2
  while (t <= T) {
    step = particle.filter.step(N, Nth, y[t], part[t-1,], w[t-1,], y[t-1], t, param)
    part[t,] = step$particles
    w[t,] = step$weights
    t = t + 1
  }
  res.list = list("particles"=part, "weights"=w)
  return(res.list)
}

particle.smoother = function(wfilt, part, param=NULL) {
  # Particle smoother (run after particle filter).
  T = length(part[,1])
  N = length(part[1,])
  one.row = matrix(1, 1, N)
  w2smooth = array(0, c(T-1, N, N))
  wsmooth = matrix(0, T, N)
  kernel = matrix(0, N, N)
  wsmooth[T,] = wfilt[T,]
  for (t in (T-1):1) {
    kernel = exp(kernel.log.pdf(part[t,], part[t+1,], t+1, t.index, param))
    wsmooth[t,] = wfilt[t,] * t(kernel %*% t(wsmooth[t+1,] / (wfilt[t,] %*% kernel)))
    w2smooth[t,,] = (wfilt[t,] %*% t(wsmooth[t+1,]) * kernel) /
      (t(one.row) %*% (wfilt[t,] %*% kernel))
  }
  res.list = list("weights"=wsmooth, "weights2"=w2smooth)
  return(res.list)
}

particle.filter.init = function(N, y, param=NULL) {
  # Particle filter initialization.
  part = generate.init(N, param)
  wei = exp(observation.log.pdf(y, part, 1, param))
  w = wei / sum(wei)
  res.list = list("particles"=part, "weights"=w)
  return(res.list)
}

particle.filter.step = function(N, Nth, y, part, w, yprev=NULL, t.index=NULL,
                                param=NULL) {
  # One particle filter step.
  Neff = 1 / sum(w**2)
  if (Neff <= Nth) {
    resampidx = sample.int(N, N, replace=TRUE, prob=w)
    part = generate.kernel(part[resampidx], yprev, t.index, param)
    wei = exp(observation.log.pdf(y, part, t.index, param))
  } else {
    part = generate.kernel(part, yprev, t.index, param)
    wei = w * exp(observation.log.pdf(y, part, t.index, param))
  }
  tryCatch(
    {w = wei / sum(wei)},
    error = function(e) {print('Particle filter | error')}
  )
  if (any(is.infinite(w)) || any(is.nan(w))) {
    w = wprev
  }
  res.list = list("particles"=part, "weights"=w)
  return(res.list)
}

particle.forecast = function(N, part, w, h, yprev, t.index=NULL, param=NULL) {
  # Particle forecast (after particle filter)
  resampidx = sample.int(N, N, replace=TRUE, prob=w)
  part = part[resampidx]
  for (t in 1:h) {
    part = generate.kernel(part, yprev, t.index + t, param)
  }
  resampidx = sample.int(N, N, replace=TRUE, prob=w)
  y.part = generate.observation(N, part[resampidx], t.index + h, param)
  return(y.part)
}

compute.PIT = function(y, y.part) {
  # Probability Integral Transform.
  PIT = sum(y.part < y) / length(y.part)
  return(PIT)
}

run.experiment = function(N, Nth, T, h, param=NULL) {
  # Run experiment with simulated data.

  x = numeric(T)
  y = numeric(T)
  y.part = matrix(NA, T, N)

  ## initialization ##

  # observe
  obs.init = observation.init(param)
  x[1] = obs.init$state
  y[1] = obs.init$observation

  # filter
  part.init = particle.filter.init(N, y[1], param)
  part = part.init$particles
  w = part.init$weights

  ## time loop ##

  t = 1
  s = h + 1
  while (t <= T - h) {

    # forecast
    y.part[s,] = particle.forecast(N, part, w, h, y[t], t, param)

    t = t + 1
    s = s + 1

    # observe
    obs.step = observation.step(x[t-1], y[t-1], t, param)
    x[t] = obs.step$state
    y[t] = obs.step$observation

    # one-step filter
    part.step = particle.filter.step(N, Nth, y[t], part, w, y[t-1], t, param)
    part = part.step$particles
    w = part.step$weights

  }

  res.list = list("observations"=y, "observation.particles"=y.part)
  return(res.list)

}

run.experiment.SV = function(N, Nth, h, param.init, param.inf, param.sup,
                             T.train, y, maxiter.init, maxiter, tol, fit.period) {
  # Run experiment with real data.

  T = length(y)
  t = T.train

  # initial model fit
  print('model initial fit')
  em = EM.algo(y[1:t], param.init, param.inf, param.sup, N, Nth, maxiter.init, tol)
  param = em$param
  last.fit = 0

  # initial particle filter
  filter = particle.filter(N, Nth, y[1:t], param)
  part = filter$particles[t,]
  w = filter$weights[t,]

  # initial forecast
  y.part = particle.forecast(N, part, w, h, y[1:t], NULL, param)

  while (t < T - h) {

    t = t + 1
    last.fit = last.fit + 1
    print(sprintf('time %i', t))

    # re-fit model
    if (last.fit == fit.period) {
      print('model re-fit')
      em = EM.algo(y[(t-T.train):t], param, param.inf, param.sup, N, Nth, maxiter, tol)
      param = em$param
      last.fit = 0
    }

    # observe new datapoint & particle filter step
    filter = particle.filter.step(N, Nth, y[t], part, w, y[t-1], NULL, param)
    part = filter$particles
    w = filter$weights

    # forecast
    y.part = rbind(y.part, particle.forecast(N, part, w, h, y[t], NULL, param))

  }

  return(y.part)
}

forecast.statistics = function(y, y.part) {
  "Compute statistics to assess distribution forecast performance."
  N = ncol(y.part)
  PIT = rowSums(y.part < y %*% matrix(1, 1, N)) / N
  probas.inf = c(.01, 0.025, 0.05)
  probas.sup = c(.95, 0.975, 0.99)
  VaR.inf = matrix(NA, length(y), length(probas.inf))
  VaR.sup = matrix(NA, length(y), length(probas.sup))
  ES.inf = matrix(NA, length(y), length(probas.inf))
  ES.sup = matrix(NA, length(y), length(probas.sup))
  for (i in 1:length(y)) {
    VaR.inf[i,] = quantile(-y.part[i,], probs=probas.inf)
    VaR.sup[i,] = quantile(-y.part[i,], probs=probas.sup)
    for (j in 1:length(probas.inf)) {
      ES.inf[i,j] = -mean(y.part[i, -y.part[i,] < VaR.inf[i,j]])
    }
    for (j in 1:length(probas.sup)) {
      ES.sup[i,j] = -mean(y.part[i, -y.part[i,] > VaR.sup[i,j]])
    }
  }
  res.list = list("PIT"=PIT,
                  "VaR.01"=VaR.inf[,1], "VaR.025"=VaR.inf[,2],  "VaR.05"=VaR.inf[,3],
                  "VaR.95"=VaR.sup[,1], "VaR.975"=VaR.sup[,2], "VaR.99"=VaR.sup[,3],
                  "ES.01"=ES.inf[,1], "ES.025"=ES.inf[,2], "ES.05"=ES.inf[,3],
                  "ES.95"=ES.sup[,1], "ES.975"=ES.sup[,2], "ES.99"=ES.sup[,3])
  return(res.list)
}
