Usage:
glm.nb [response file] [predictor file] [initial theta]

Inputs:
[response file]: A text file containing the responses (y vector) for each observation, one per line.

[predictor file]: A text file containing the feature values (x matrix) to use in the negative binomial fitting, where rows are observations, and columns are features.

[initial theta]: Used to determine convergence of the underlying algorithms. In R, this is typically set to 0.000001

Output:
"glm.nb" prints to standard output, which can be redirected. This output includes:
- Coefficients
- Residuals
- Fitted Values
- Effects
- R (correlation measure)
- Rank
- QR matrix
- Qraux
- Pivot vector
- Tol
- Linear predictors
- Deviance
- AIC
- Null deviance
- Number of iterations
- Weights
- Prior weights
- Degrees of freedom residual
- Degrees of freedom null
- Converged
- Boundary
- Theta
- SE Theta
- Two Log Likelihood
