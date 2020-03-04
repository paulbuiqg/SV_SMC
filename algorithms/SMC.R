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
  return(list("particles"=part, "weights"=w))
}

particle.filter.init = function(N, y, param) {
  # Particle filter initialization.
  part = generate.init(N, param)
  wei = exp(observation.log.pdf(y, part, 1, param))
  w = normalize.weights(wei, rep(1/N, N))  
  res.list = list("particles"=part, "weights"=w)
  return(res.list)
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

info.particle.filter = function(N, Nth, y, param) {
  # Particle filter with computation of score and observed information.
  T = length(y)
  d = length(param)
  part = matrix(0, T, N)
  w = matrix(0, T, N)
  score = matrix(0, T, d)
  info = array(0, c(T, d, d))
  init = info.particle.filter.init(N, y[1], param)
  part[1,] = init$particles
  w[1,] = init$weights
  alpha = init$alpha
  beta = init$beta
  scorinfo = score.info(w[1,], alpha, beta)
  score[1,] = scorinfo$score
  info[1,,] = scorinfo$info
  t = 2
  while (t <= T) {
    step = info.particle.filter.step(N, Nth, y[t], part[t-1,], w[t-1,], alpha, beta, y[t-1], t, param)
    part[t,] = step$particles
    w[t,] = step$weights
    alpha = step$alphas
    beta = step$betas
    scorinfo = score.info(w[t,], alpha, beta)
    score[t,] = scorinfo$score
    info[t,,] = scorinfo$info
    t = t + 1
  }
  return(list("particles"=part, "weights"=w, "scores"=score, "infos"=info))
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

score.particle.filter = function(N, Nth, y, param) {
  # Particle filter with computation of score.
  T = length(y)
  d = length(param)
  part = matrix(0, T, N)
  w = matrix(0, T, N)
  score = array(0, c(T, d))
  init = score.particle.filter.init(N, y[1], param)
  part[1,] = init$particles
  w[1,] = init$weights
  alpha = init$alpha
  score[1,] = w[1,] %*% alpha
  t = 2
  while (t <= T) {
    step = score.particle.filter.step(N, Nth, y[t], part[t-1,], w[t-1,], alpha, y[t-1], t, param)
    part[t,] = step$particles
    w[t,] = step$weights
    alpha = step$alphas
    score[t,] = w[t,] %*% alpha
    t = t + 1
  }
  return(list("particles"=part, "weights"=w, "scores"=score))
}

score.particle.filter.init = function(N, y, param) {
  # Initialization of particle filter with score.
  d = length(param)
  part = generate.init(N, param)
  wei = exp(observation.log.pdf(y, part, 1, param))
  alpha = deriv.observation.log.pdf(y, part, 1, param) + deriv.init.log.pdf(part, param)
  w = normalize.weights(wei, rep(1/N, N))
  return(list("particles"=part, "weights"=w, "alphas"=alpha))
}

score.particle.filter.step = function(N, Nth, y, partprev, wprev, alphaprev, yprev, t.index, param) {
  # One step of particle filter with score.
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
  alpha = matrix(0, N, d)
  kernel = exp(kernel.log.pdf(partprev, part, NULL, t.index, param))
  for (i in 1:N) {
    alpha[i,] = colSums(((wprev * kernel[,i]) %*% matrix(1, 1, d)) * 
      (matrix(1, N, 1) %*% deriv.observation.log.pdf(y, part[i], t.index, param) +
         deriv.kernel.log.pdf(partprev, part[i], NULL, t.index, param) + alphaprev)) /
           c(wprev %*% kernel[,i])
  }
  return(list("particles"=part, "weights"=w, "alphas"=alpha))
}

compute.score.info = function(weights, alphas, betas) {
  # Compute score and observed information.
  N = length(weights)
  score = t(weights) %*% alphas
  info = t(score) %*% score
  for (i in 1:N) {
    info = info + weights[i] * (alphas[i,] %*% t(alphas[i,]) + betas[i,,])
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