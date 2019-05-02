rm(list=ls())

library(GAS)

set.seed(99)

path = '/home/paul/code/SMC/'

# read data
y = StockIndices[,1]
T = length(y)
t = floor(0.8 * T) # in-sample data size

# load EM & SMC functions
source(paste(path, 'SMC.R', sep=''))
source(paste(path, 'EM.R', sep=''))

# load model
source(paste(path, 'model_SV.R', sep=''))

# algorithm parameters
N = 50
Nth = N # avoid zero weight
tol = 1e-3
fit.period = 60
h = 1
y.part = matrix(NA, T, N)

# initial model fit
print(sprintf('- time index: %i', t))
em = EM.algo(y[1:t], param.init, param.inf, param.sup, N, Nth, 20, tol)
param = em$param
last.fit = 0

# initial particle filter
filter = particle.filter(N, Nth, y[1:t], param)
part = filter$particles[t,]
w = filter$weights[t,]

# initial forecast
y.part[t+h,] = particle.forecast(N, part, w, h, y[t], NULL, param)

while (t < T - h) {
  
  t = t + 1
  last.fit = last.fit + 1
  print(sprintf('- time index: %i', t))
  
  # re-fit model
  if (last.fit == fit.period) {
    em = EM.algo(y[1:t], param, param.inf, param.sup, N, Nth, 10, tol)
    param = em$param
    last.fit = 0
  }
  
  # observe new datapoint & particle filter step
  filter = particle.filter.step(N, Nth, y[t], part, w, y[t-1], NULL, param)
  part = filter$particles
  w = filter$weights
  
  # forecast
  y.part[t+h,] = particle.forecast(N, part, w, h, y[t], NULL, param)
  
}