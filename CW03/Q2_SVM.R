## Question 2 ----
# load libraries ----
library(mvtnorm)
library(dplyr)
library(ggplot2)
library(caret)
library(pROC)

# simulate data ----
# define a function to generate Gaussian distributed data
generate.gaussian <- function(n, center, sigma, label) {
  data <- rmvnorm(n, mean = center, sigma = sigma)   # random number generator
  data <- data.frame(data)
  names(data) <- c('dim1', 'dim2')   # label 2 dimensions
  data <- data %>% mutate(class=factor(label))   # create a new column named 'class'
  data   # return a data frame
}
# define a function to generate symmetric matrix
sym.matrix <- function(nrow, diag, off.diag) {
  n <- nrow   # number of rows
  d <- diag   # diagonal elements
  od <- off.diag   # off-diagonal elements
  matrix <- matrix(rep(d, n*n), nrow = n)   # initialise a matrix
  matrix[upper.tri(matrix)] <- od   # assign the upper triangle of matrix
  matrix[lower.tri(matrix)] <- od   # assign the lower triangle of matrix
  matrix
}
# create 3 classes
n1 <- 50   # number of obs in class 1
n2 <- 50   # number of obs in class 2
n3 <- 50   # number of obs in class 3
# class 1 and 2 - a mixture of Gaussian distributions
# parameters for multivariate normal distribution
center1 <- c(-1,0)   # mean vector of class 1
center2 <- c(1,0)   # mean vector of class 2
sigma1 <- sym.matrix(nrow = 2, diag = 0.5, off.diag = 0.1)   # covariance matrix of class 1
sigma2 <- sym.matrix(nrow = 2, diag = 0.5, off.diag = 0.1)   # covariance matrix of class 2
# generate class 1 and 2
set.seed(123)   # for reproducibility
data1 <- generate.gaussian(n1, center1, sigma1, 'Class1')   # generate random values for class 1
data2 <- generate.gaussian(n2, center2, sigma2, 'Class2')   # generate random values for class 2
# class 3 - random values on a circle
radius <- 3   # radius of the circle
x <- 0; y <- 0   # center of the circle
alpha <- 2 * pi * runif(n3)   # random angle
x <- radius * cos(alpha) + x   # dimension 1
y <- radius * sin(alpha) + y   # dimension 2
data3 <- data.frame(dim1 = x, dim2 = y)
data3 <- data3 %>% mutate(class=factor('Class3'))   # create a new column named 'class'

# generate one simulation data set
data <- bind_rows(data1, data2, data3)   # stack the 3 data frames
# have a look at data set
data %>% ggplot(aes(x=dim1, y=dim2, shape=class, colour=class)) +   # visual properties
  geom_point() +    # show scatter plots
  coord_fixed() +   # the same scale of x and y
  scale_shape_manual(values=c(1,2,3)) +  # symbols for points
  scale_colour_manual(values=c('#CC0000','#009E73','#0072B2'))   # colours for points

# random split ----
set.seed(12)   # for reproducibility
# create index for 50% training set
idx.train <- createDataPartition(data$class, 
                                 p = 0.5, times = 1, list = FALSE)
train.feature <- data[idx.train,-3]   # training features
train.label <- data$class[idx.train]   # training labels
test.feature <- data[-idx.train,-3]   # test features
test.label <- data$class[-idx.train]   # test labels

# train SVM with polynomial kernel ----
# set up training control
fitControl <- trainControl(method = 'repeatedcv',   
                           number = 5,   # 5-fold cross-validation
                           repeats = 3,   # cv repeated 3 times
                           classProbs = TRUE)   # get probabilities
# set up a grid of tuning parameters
grid.poly <- expand.grid(degree = c(2, 3, 4, 5),   # polynomial degree
                         scale = c(0.01, 0.1, 1),   # scale
                         C = c(0.01, 0.1, 1, 10))   # cost
# training process
set.seed(122)
svm.poly <- train(x = train.feature, y = train.label,
                  method = 'svmPoly',   # SVM with Polynomial Kernel (kernlab)
                  trControl = fitControl,
                  preProcess = c('center', 'scale'),
                  tuneGrid = grid.poly)
# have a look at output
svm.poly
# plot training accuracy against parameters
plot(svm.poly)
# compute test error
pred.poly <- predict(svm.poly, test.feature)
err.poly <- mean(pred.poly!=test.label)
err.poly

# train SVM with RBF kernel ----
# set up training control
fitControl <- trainControl(method = 'repeatedcv',   
                           number = 5,   # 5-fold cross-validation
                           repeats = 3,   # cv repeated 3 times
                           classProbs = TRUE)   # get probabilities
# set up a grid of tuning parameters
grid.radial <- expand.grid(sigma = c(0.01, 0.1, 1, 2, 4, 6, 8, 10),   # gamma
                           C = c(0.01, 0.1, 1, 10))   # cost
# training process
set.seed(122)
svm.radial <- train(x = train.feature, y = train.label,
                    method = 'svmRadial',   # SVM with Radial Basis Function Kernel (kernlab)
                    trControl = fitControl,
                    preProcess = c('center', 'scale'),
                    tuneGrid = grid.radial)
# have a look at output
svm.radial
# plot training accuracy against parameters
plot(svm.radial)
# compute test error
pred.radial <- predict(svm.radial, test.feature)
err.radial <- mean(pred.radial!=test.label)
err.radial

# train SVC ----
# set up training control
fitControl <- trainControl(method = 'repeatedcv',   
                           number = 5,   # 5-fold cross-validation
                           repeats = 3,   # cv repeated 3 times
                           classProbs = TRUE)   # get probabilities
# set up a grid of tuning parameters
grid.linear <- expand.grid(C = c(0.01, 0.1, 1, 10, 20, 30, 40, 50))   # cost
# training process
set.seed(122)
svm.linear <- train(x = train.feature, y = train.label,
                    method = 'svmLinear',   # SVM with Linear Kernel (kernlab)
                    trControl = fitControl,
                    preProcess = c('center', 'scale'),
                    tuneGrid = grid.linear)  
# have a look at output
svm.linear
# plot training accuracy against parameters
plot(svm.linear)
# compute test error
pred.linear <- predict(svm.linear, test.feature)
err.linear <- mean(pred.linear!=test.label)
err.linear

# compare test errors ----
err <- data.frame(err.poly, err.radial, err.linear)
rownames(err) <- 'test error rate'
colnames(err) <- c('polynomial', 'RBF', 'linear')
err

# calculate AUC values ----
# compute AUC of polynomial kernel
prob.poly <- predict(svm.poly, test.feature, type = 'prob')
roc.poly <- multiclass.roc(predictor = prob.poly, response = test.label)
auc.poly <- roc.poly$auc
auc.poly
# compute AUC of polynomial kernel
prob.radial <- predict(svm.radial, test.feature, type = 'prob')
roc.radial <- multiclass.roc(predictor = prob.radial, response = test.label)
auc.radial <- roc.radial$auc
auc.radial
# compute AUC of polynomial kernel
prob.linear <- predict(svm.linear, test.feature, type = 'prob')
roc.linear <- multiclass.roc(predictor = prob.linear, response = test.label)
auc.linear <- roc.linear$auc
auc.linear
# compare the results
auc <- data.frame(auc.poly, auc.radial, auc.linear)
rownames(auc) <- '3-class AUC (test)'
colnames(auc) <- c('polynomial', 'RBF', 'linear')
auc


