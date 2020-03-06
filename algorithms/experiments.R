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

run.experiment.sv = function(N, Nth, h, T.train, y, param.init, param.inf, param.sup,
                             maxiter.init, maxiter, tol, fit.period) {
  # Run experiment with real data.
  
  T = length(y)
  t = T.train
  
  # initial model fit
  print('model initial fit')
  est = EM.algo(y[1:t], param.init, param.inf, param.sup, N, Nth, maxiter.init, tol)
  new.param = est$param
  last.fit = 0
  
  # initial particle filter
  filter = particle.filter(N, Nth, y[1:t], new.param)
  part = filter$particles[t,]
  w = filter$weights[t,]
  
  # initial forecast
  y.part = particle.forecast(N, part, w, h, y[1:t], NULL, new.param)
  
  # initial parameter
  params = new.param
  
  while (t < T - h) {
    
    t = t + 1
    print(sprintf('time %i', t))
    
    # re-fit model
    if (last.fit == fit.period) {
      print('model re-fit')
      est = EM.algo(y[(t-T.train):t], new.param, param.inf, param.sup, N, Nth, maxiter, tol)
      new.param = est$param
      last.fit = 0
    } else {
      last.fit = last.fit + 1
    }
    
    # observe new datapoint & particle filter step
    filter = particle.filter.step(N, Nth, y[t], part, w, y[t-1], NULL, new.param)
    part = filter$particles
    w = filter$weights
    
    # forecast
    y.part = rbind(y.part, particle.forecast(N, part, w, h, y[t], NULL, new.param))
    
    # parameter
    params = rbind(params, new.param)
    
  }
  
  return(list("observation.particles"=y.part, "param.seq"=param.seq))
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