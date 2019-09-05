rm(list=ls())

set.seed(99)

path = '/home/paul/code/SV_SMC/'
path.model = paste(path, 'models', sep='')
path.algo = paste(path, 'algorithms', sep='')

source(paste(path.model, 'bench.R', sep='/'))
source(paste(path.algo, 'SMC.R', sep='/'))
source(paste(path.algo, 'EM.R', sep='/'))

n.xp = 100
T = 1000

N.grid = c(50, 100, 200, 400)
h = 5

box.pvals = matrix(NA, length(N.grid), n.xp)
box.stats = matrix(NA, length(N.grid), n.xp)
ks.pvals = matrix(NA, length(N.grid), n.xp)
ks.stats = matrix(NA, length(N.grid), n.xp)
for (i in 1:length(N.grid)) {
  N = N.grid[i]
  print(sprintf('N=%i', N))
  for (j in 1:n.xp) {
    xp = run.experiment(N, N, T, h, param)
    y = xp$observations
    y.part = xp$observation.particles
    PITs = rowSums(y.part < y %*% matrix(1, 1, N)) / N
    PITs.sub = PITs[seq(h+1, T, h)]
    box = Box.test(PITs.sub, type='Ljung-Box')
    ks = ks.test(PITs.sub, punif)
    box.pvals[i,j] = box$p.value
    box.stats[i,j] = box$statistic
    ks.pvals[i,j] = ks$p.value
    ks.stats[i,j] = ks$statistic
  }
}

n = length(PITs.sub)

par(mfrow=c(2, 2), oma=c(0, 0, 2, 0))
xgrid = seq(-1, 10, .01)
plot(ecdf(box.stats[1,]), xlim=c(-1, 10),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[1]))
lines(xgrid, pchisq(xgrid, 1))
plot(ecdf(box.stats[2,]), xlim=c(-1, 10),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[2]))
lines(xgrid, pchisq(xgrid, 1))
plot(ecdf(box.stats[3,]), xlim=c(-1, 10),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[3]))
lines(xgrid, pchisq(xgrid, 1))
plot(ecdf(box.stats[4,]), xlim=c(-1, 10),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[4]))
lines(xgrid, pchisq(xgrid, 1))
mtext(sprintf('Ljung-Box test - %i-step ahead forecast', h), outer=TRUE)

library(kolmim)
par(mfrow=c(2, 2))
xgrid = seq(.001, .3, .001)
ygrid = numeric(length(xgrid))
for (i in 1:length(ygrid)) {ygrid[i] = pkolm(xgrid[i], n)}
plot(ecdf(ks.stats[1,]), xlim=c(min(xgrid), max(xgrid)),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[1]))
lines(xgrid, ygrid)
plot(ecdf(ks.stats[2,]), xlim=c(min(xgrid), max(xgrid)),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[2]))
lines(xgrid, ygrid)
plot(ecdf(ks.stats[3,]), xlim=c(min(xgrid), max(xgrid)),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[3]))
lines(xgrid, ygrid)
plot(ecdf(ks.stats[4,]), xlim=c(min(xgrid), max(xgrid)),
     xlab='Test statistic', ylab='ECDF', main=sprintf('N=%i', N.grid[4]))
lines(xgrid, ygrid)
mtext(sprintf('Kolmogorov-Smirnov test - %i-step ahead forecast', h), outer=TRUE)