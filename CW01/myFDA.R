myFDA <- function(X,y){
    ########################################################
    # This function calculates the linear discriminant for binary
    # classification.
    # Input: Feature matrix, X (N by p) and label vector, y (N by 1)
    # Output: Linear discriminant, w (p by 1)
    ########################################################
    k <- 2   # for binary classification task
    n <- nrow(X)   # total number of observations
    # get class names
    name1 <- levels(factor(y))[1]
    name2 <- levels(factor(y))[2]
    # boolean values for each class
    bool.k1 <- (y == name1)
    bool.k2 <- (y == name2)
  
    # divide features into 2 groups according to labels
    class1 <- X[bool.k1,]
    class2 <- X[bool.k2,]
    # number of observations within each class
    n1 <- nrow(class1)
    n2 <- nrow(class2)
    # get covariance matrix for each group
    cov1 <- cov(class1)
    cov2 <- cov(class2)
    
    # get the within-class scatter
    sigma <- 1/(n-k) * (cov1*(n1-1) + cov2*(n2-1))   # using covariance matrix
    sigma.inv <- solve(sigma)   # inverse of matrix
    
    # calculate mean difference between classes for each feature
    mu <- data.frame(mu1 = matrix(apply(class1, 2, mean)),   # feature means of class 1
                     mu2 = matrix(apply(class2, 2, mean)))   # feature means of class 2
    mu.diff <- matrix(mu[,2] - mu[,1])
    
    # (mu2-mu1) is normalised by the within-class scatter
    coef <- sigma.inv %*% mu.diff
    scalar <- sqrt(t(coef) %*% sigma %*% coef)
    w <- coef/drop(scalar)  # get the vector of discriminant coefficients
    
    return(w)
}
