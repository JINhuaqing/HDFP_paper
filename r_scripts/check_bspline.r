# check whether the bsplines from two fns are the same

library(fda)
#?create.bspline.basis
library(splines)
#bs

iknots <- c(0.2, 0.4, 0.5)
bknots <- c(0, 1)
x <- seq(0, 1, length.out = 100)

bsp1 <- create.bspline.basis(breaks=sort(c(iknots, bknots)))
basismat1 <- eval.basis(x, bsp1)

basismat2 <- bs(x, degree=3, knots=iknots, Boundary.knots = bknots, intercept=1)


basismat1 - basismat2

