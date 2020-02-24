generate.series = function(T, param) {
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

observation.init = function(param) {
  # Generate initial state & observation.
  x = generate.init(1, param)
  y = generate.observation(1, x[1], 1, param)
  res.list = list("state" = x, "observation" = y)
  return(res.list)
}

observation.step = function(xprev, yprev, t.index, param) {
  # Generate next state & observation.
  x = generate.kernel(xprev, yprev, t.index, param)
  y = generate.observation(1, x, t.index, param)
  res.list = list("state" = x, "observation" = y)
  return(res.list)
}

particle.filter = function(N, Nth, y, param) {
  # Generic particle filter.
  T = length(y)
  part = matrix(0, T, N)
  w = matrix(0, T, N)
  init = particle.filter.init(N, y[1], param)
  part[1,] = init$particles
  w[1,] = init$weights
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

info.particle.filter = function(N, Nth, y, param) {
  # Particle filter with computation of score and observed information.
  T = length(y)
  d = length(param)
  score = array(0, c(T, d))
  info = array(0, c(T, d, d))
  init = info.particle.filter.init(N, y[1], param)
  part = init$particles
  w = init$weights
  alpha = init$alpha
  beta = init$beta
  scorinfo = score.info(w, alpha, beta)
  score[1,] = scorinfo$score
  info[1,,] = scorinfo$info
  t = 2
  while (t <= T) {
    step = info.particle.filter.step(N, Nth, y[t], part, w, alpha, beta, y[t-1], t, param)
    part = step$particles
    w = step$weights
    alpha = step$alphas
    beta = step$betas
    scorinfo = score.info(w, alpha, beta)
    score[t,] = scorinfo$score
    info[t,,] = scorinfo$info
    t = t + 1
  }
  return(list("scores"=score, "infos"=info))
}

particle.filter.init = function(N, y, param) {
  # Particle filter initialization.
  part = generate.init(N, param)
  wei = exp(observation.log.pdf(y, part, 1, param))
  w = normalize.weights(wei, rep(1/N, N))  
  res.list = list("particles"=part, "weights"=w)
  return(res.list)
}
  
info.particle.filter.init = function(N, y, param) {
  # Initialization of particle filter with score and observed information.
  d = length(param)
  part = generate.init(N, param)
  wei = exp(observation.log.pdf(y, part, 1, param))
  alpha = array(0, c(N, d))
  beta = array(0, c(N, d, d))
  for (i in 1:N) {
    alpha[i,] = deriv.observation.log.pdf(y, part[i], 1, param) + deriv.init.log.pdf(part[i], param)
    beta[i,,] = deriv2.observation.log.pdf(y, part[i], 1, param) + deriv2.init.log.pdf(part[i], param)
  }
  w = normalize.weights(wei, rep(1/N, N))  
  return(list("particles"=part, "weights"=w, "alphas"=alpha, "betas"=beta))
}
 
particle.filter.step = function(N, Nth, y, part, w, yprev, t.index, param) {
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
  w = normalize.weights(wei, w)
  res.list = list("particles"=part, "weights"=w)
  return(res.list)
}

info.particle.filter.step = function(N, Nth, y, partprev, wprev, alphaprev, betaprev, yprev, t.index, param) {
  # One step of particle filter with score and observed information.
  d = length(param)
  Neff = 1 / sum(wprev**2)
  if (Neff <= Nth) {
    resampidx = sample.int(N, N, replace=TRUE, prob=wprev)
    part = generate.kernel(partprev[resampidx], yprev, t.index, param)
    wei = exp(observation.log.pdf(y, part, t.index, param))
  } else {
    part = generate.kernel(partprev, yprev, t.index, param)
    wei = wprev * exp(observation.log.pdf(y, part, t.index, param))
  }
  w = normalize.weights(wei, w)
  alpha = array(0, c(N, d))
  beta = array(0, c(N, d, d))
  for (i in 1:N) {
    constant = numeric(N)
	alpha[i,] = numeric(d)
	beta[i,,] = matrix(0, d, d)
    for (j in 1:N) {
  	  constant[j] = wprev[j] * exp(kernel.log.pdf(partprev[j], part[i], NULL, t.index, param))
      deriv = deriv.observation.log.pdf(y, part[i], t.index, param) +
	    deriv.kernel.log.pdf(partprev[j], part[i], NULL, t.index, param) +
		  alphaprev[j,]
      deriv2 = deriv2.observation.log.pdf(y, part[i], t.index, param) +
	    deriv2.kernel.log.pdf(partprev[j], part[i], NULL, t.index, param) +
		  betaprev[j,,]
	  alpha[i,] = alpha[i,] + constant[j] * deriv
	  beta[i,,] = beta[i,,] + constant[j] * (deriv %*% t(deriv) + deriv2)
    }
    alpha[i,] = alpha[i,] / sum(constant)
	beta[i,,] = beta[i,,] / sum(constant) - alpha[i,] %*% t(alpha[i,])
  }
  return(list("particles"=part, "weights"=w, "alphas"=alpha, "betas"=beta))
}

score.info = function(w, alpha, beta) {
  # Compute score and observed information.
  N = length(w)
  score = t(w) %*% alpha
  info = t(score) %*% score
  for (i in 1:N) {
    info = info + w[i] * (alpha[i,] %*% t(alpha[i,]) + beta[i,,])
  }
  return(list("score"=score, "info"=info))
}

normalize.weights = function(wei, wprev) {
  # Normalize weights.
  if (any(is.infinite(wei)) || any(is.nan(wei))) {
    print('Particle filter | error | NaN or Inf weight')
    return(wprev)
  }
  tryCatch(
    {
      w = wei / sum(wei)
    },
    error = function(e) {print('Particle filter | error | weight normalization'); return(wprev)}
  )
  if (any(is.infinite(w)) || any(is.nan(w))) {
    print('Particle filter | error | NaN or Inf weight')
    return(wprev)
  } else {
    return(w)
  }
}

particle.smoother = function(wfilt, part, y, param) {
  # Particle smoother (run after particle filter).
  T = length(part[,1])
  N = length(part[1,])
  one.row = matrix(1, 1, N)
  w2smooth = array(0, c(T-1, N, N))
  wsmooth = matrix(0, T, N)
  kernel = matrix(0, N, N)
  wsmooth[T,] = wfilt[T,]
  for (t in (T-1):1) {
    kernel = exp(kernel.log.pdf(part[t,], part[t+1,], y[t], t+1, param))
    wsmooth[t,] = wfilt[t,] * t(kernel %*% t(wsmooth[t+1,] / (wfilt[t,] %*% kernel)))
    w2smooth[t,,] = (wfilt[t,] %*% t(wsmooth[t+1,]) * kernel) /
      (t(one.row) %*% (wfilt[t,] %*% kernel))
  }
  res.list = list("weights"=wsmooth, "weights2"=w2smooth)
  return(res.list)
}

particle.forecast = function(N, part, w, h, yprev, t.index, param) {
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

run.experiment = function(N, Nth, T, h, param) {
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

run.experiment.SV = function(N, Nth, h, param.init, param.inf, param.sup, T.train, y, maxiter.init, maxiter, tol, fit.period) {
  # Run experiment with real data.

  T = length(y)
  t = T.train

  # initial model fit
  print('model initial fit')
  em = EM.algo(y[1:t], param.init, param.inf, param.sup, N, Nth, maxiter.init, tol)
  param = em$param
  last.fit = 0
  
  # # parameter estimates and their std 
  # info.filter = info.particle.filter(N, Nth, y, param)
  # info = info.filter$infos[t,,]

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
  # Compute statistics to assess distribution forecast performance.
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