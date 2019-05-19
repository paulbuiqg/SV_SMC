# SPOT – are the Spot prices
# SPOT.dRet – are daily returns in the period 1997-2017
# SPOT.wRet – are weekly returns in the period 1997-2017
# SPOT.mRet – are monthly returns in the period 1997-2017
# 
# In total we have nine assets (as in Chan, Grant, 2016):
# WTI, Brent, New York Harbor Gasoline, U.S. Gulf Gasoline, New York Harbor Heating Oil No. 2,
# Los Angeles Diesel, U.S. Gulf Kerosine, Propane, Henry Hub Natural Gas.
# 
# Take a three year rolling training data set to forecast the 1-day, 1-week
# and 1-month ahead Value-at-Risk and Expected Shortfall from 2000 to 2017 
# using the daily, weekly, and monthly returns, respectively
# (i.e. daily returns for 1-day ahead, weekly returns for 1-week ahead, ...).
# 
# Important is that we keep track of the actual date of the forecast,
# so we can compare the forecasts in a given year.
# 
# Value-at-Risk and Expected Shortfall are needed for 1%, 2.5%, 5%, 95%, 97.5% and 99%,
# where Value-at-Risk is the quantile and Expected Shortfall is the mean of the returns exceeding the quantile.

rm(list=ls())
set.seed(99)
library(zoo)

### settings ###

path = '/home/paul/code/SV_SMC/'
model = 'SV'
freq = 'week'

# out-of-sample starting time
start.out = as.POSIXct('2000-01-01', origin='1970-01-01', tz='GMT')

# length of training data
len.train = as.difftime(1096, units='days')

# parameters
tol = 0.001
N = 300
Nth = N

### load code & data ###

path.model = paste(path, 'models/', sep='')
path.algo = paste(path, 'algorithms/', sep='')
path.data = paste(path, 'data/', sep='')

# EM & SMC function loading
source(paste(path.algo, 'SMC.R', sep=''))
source(paste(path.algo, 'EM.R', sep=''))

# model loading
source(paste(path.model, model, '.R', sep=''))

# data loading
load(paste(path.data, 'EnergyReturns.RData', sep=''))

###

# frequency-dependent settings
if (freq=='day') {
  fit.period = 60
  maxiter.init = 200
  maxiter = 20
  assign('data', SPOT.dRet)
} else if (freq=='week') {
  fit.period = 30
  maxiter.init = 100
  maxiter = 10
  assign('data', SPOT.wRet)
} else if (freq=='month') {
  fit.period = 3
  maxiter.init = 50
  maxiter = 5
  assign('data', SPOT.mRet)
}

# missing values processing
data = na.trim(data, sides='both')
data = na.locf(data)

# in-sample starting time
start.in = max(start.out - len.train, min(index(data)))
data.0 = data[index(data) >= start.in]

forecast.index = index(data.0)[index(data.0) >= start.out]
T.train = length(index(data.0)[index(data.0) < start.out])

for (j in 1:ncol(data)) {

  # model fitting, particle forecasting
  series = coredata(data.0[,j])
  param.init = initialize.parameters(series)
  series.forecast = run.experiment.SV(N, Nth, 1, param.init, param.inf, param.sup,
                                     T.train, series, maxiter.init, maxiter, tol, fit.period)
  
  # PIT, VaR, ES
  stats = forecast.statistics(series[index(data.0) >= min(forecast.index)], series.forecast)
  PIT = stats$PIT
  VaR.01 = stats$VaR.01; VaR.025 = stats$VaR.025; VaR.05 = stats$VaR.05
  VaR.95 = stats$VaR.95; VaR.975 = stats$VaR.975; VaR.99 = stats$VaR.99
  ES.01 = stats$ES.01; ES.025 = stats$ES.025; ES.05 = stats$ES.05
  ES.95 = stats$ES.95; ES.975 = stats$ES.975; ES.99 = stats$ES.99
  
  # result dataframe (model x asset x frequency)
  df = data.frame(matrix(ncol=14, nrow=length(forecast.index)))
  asset.name = attr(data, 'dimnames')[[2]][j]
  colnames(df) = c(asset.name, 'PIT',
                   'VaR.01', 'VaR.025', 'VaR.05', 'VaR.95', 'VaR.975', 'VaR.99',
                   'ES.01', 'ES.025', 'ES.05', 'ES.95', 'ES.975', 'ES.99')
  row.names(df) = forecast.index
  df[asset.name] = series[(T.train+1):length(series)]; df['PIT'] = PIT
  df['VaR.01'] = VaR.01; df['VaR.025'] = VaR.025; df['VaR.05'] = VaR.05
  df['VaR.95'] = VaR.95; df['VaR.975'] = VaR.975; df['VaR.99'] = VaR.99
  df['ES.01'] = ES.01; df['ES.025'] = ES.025; df['ES.05'] = ES.05
  df['ES.95'] = ES.95; df['ES.975'] = ES.975; df['ES.99'] = ES.99
  name.df = paste(model, asset.name, freq, sep='__')
  save(df, file=paste(path.data, name.df, '.RData', sep=''))
  
}
