rm(list=ls())

set.seed(99)

path = '/home/paul/code/SV_SMC/'
path.model = paste(path, 'models/', sep='')
path.algo = paste(path, 'algorithms/', sep='')

source(paste(path.model, 'SV.R', sep=''))
source(paste(path.algo, 'SMC.R', sep=''))
source(paste(path.algo, 'EM.R', sep=''))

## simulate data ##

series = generate.series(3000, param.test)
y = series$observations
x = series$states

plot(y, type="l")

## test SMC ##

# filter = particle.filter(100, 66, y, param.test)
# filter = info.particle.filter(100, 66, y, param.test)
filter = score.particle.filter(100, 66, y, param.test)
x.filt = rowSums(filter$particles * filter$weights)
 
# smoother = particle.smoother(filter$weights, filter$particles, y, param.test)
# x.smooth = rowSums(filter$particles * smoother$weights)
 
plot(x, type="l")
lines(x.filt, col='blue')
# lines(x.smooth, col='green')

## test parameter estimation ##
# param.init = initialize.parameters(y)
# EM = EM.algo(y, param.init, param.inf, param.sup, 50, 33, 200, .001)
