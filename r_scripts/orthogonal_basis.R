rm(list=ls())
library(orthogonalsplinebasis)
library(fda)

order <- 4
knots <- seq(0, 1, 0.1)
eknots <- expand.knots(knots, order=order)

# first way
basis <- SplineBasis(eknots, order=order)

# second way
basis1 <- create.bspline.basis(breaks=knots, norder=order)

# evaluate them
xs <- seq(0, 1, 0.01)
fns1 <- eval.basis(xs, basis1)
fns <- evaluate(basis, xs)
mean((fns - fns1)**2)


# get orthogonal basis
xs <- seq(0, 1, 0.01)
obasis1 <- OrthogonalSplineBasis(eknots, order=order)
obasis2 <- OBasis(eknots, order=order)
mean((evaluate(obasis1, xs) -  evaluate(obasis2, xs))**2)

xs <- seq(0, 1, 0.001)
obasis_mat <- evaluate(obasis1, xs)
t(obasis_mat)  %*% obasis_mat/length(xs)

GramMatrix(obasis1)
