#  Import numpy
import numpy as np

#  store the variables in arrays
prob = np.array([0.25, 0.5, 0.25])
rate_1 = np.array([0.05, 0.075, 0.10])
rate_2 = np.array([0.2, 0.15, 0.1])

#  expected return of each investment
expected_return1 = np.sum(prob * rate_1)
expected_return2 = np.sum(prob * rate_2)

#  expected return of the equally weighted portfolio
weights = np.array([0.5, 0.5])
individual_returns = np.array([expected_return1, expected_return2])
portfolio_returns = np.dot(weights, individual_returns)

#  covariance matrix given probabilities
cov_matrix = np.cov(rate_1, rate_2, ddof=0, aweights=prob)

#  variance and standard deviation of each investment
var1 = cov_matrix[0,0]
var2 = cov_matrix[1,1]
std1 = np.sqrt(var1)
std2 = np.sqrt(var2)

#  correlation between Asset 1 & 2's returns
cov = cov_matrix[0,1]
corr = cov / (std1 * std2)

#  variance of portfolio
portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))

#  standard deviation (volatility of the portfolio)
portfolio_vols = np.sqrt(portfolio_var)

def percentage (number):
    return str(round(number, 4) * 100) + '%'

print('Expected Return of Investment 1 = {}'.format(percentage(expected_return1)))
print('Expected Return of Investment 2 = {}'.format(percentage(expected_return2)))
print('Expected Return of Portfolio = {}'.format(percentage(portfolio_returns)))
print('Standard Deviation of Investment 1 = {}'.format(percentage(std1)))
print('Standard Deviation of Investment 1 = {}'.format(percentage(std2)))
print('Correlation between Returns of 1 & 2 = {}'.format(round(corr, 4)))
print('Risk of Portfilio = {}'.format(percentage(portfolio_vols)))
