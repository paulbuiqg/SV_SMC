batch_parameter_estimation = function(N, Nth, y, thres, maxiter=100) {
  # Batch likelihood maximization.
  last.param = initialize.parameters(y)
  params = param
  iter = 0
  rate = thres + 1
  while (rate > thres & iter < maxiter) {
    filter = info.particle.filter(N, Nth, y, last.param)
    scorinfo = compute.score.info(filter$weights, filter$alphas, filter$betas)
    score = scorinfo$score
    info = scorinfo$info
    new.param = last.param - 0.9 * inv(info) %*% score
    params = rbind(new.param)
    rate = sqrt(sum((new.param - last.param)**2) / sum(last.param**2))
    iter = iter + 1
    last.param = new.param
  }
  return(params)
}

recursive_parameter_estimation = function(N, Nth, y, last.param, last.score, thres, maxiter=100) {
  # Recursive likelihood maximization.
  
  filter = info.particle.filter(N, Nth, y, last.param)
  scorinfo = compute.score.info(filter$weights, filter$alphas, filter$betas)
  score = scorinfo$score
  param = last.param + 0.9 * score - 0.9 * last.score
}