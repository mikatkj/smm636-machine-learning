## Question 1 ----
# load libraries and import data ----
library(caret)
library(pROC)
library(ggplot2)

thyroid <- read.table('newthyroid.txt', header = TRUE, sep=',')

# check the data
str(thyroid)
table(thyroid$class)
ratio.imbalance <- table(thyroid$class)[2]/table(thyroid$class)[1]   # calculate the ratio of imbalance
ratio.imbalance
# pre-process the data
thyroid$class <- factor(thyroid$class)   # make the class factor
# explore the boxplot
featurePlot(x = thyroid[,-1],
            y = thyroid$class,
            plot = "box",
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(5, 1),
            auto.key = list(columns = 2))

## Q.i) ----
# random splitting
tt <- 20   # repeat 20 times
set.seed(12)   # for reproducibility
train.index <- createDataPartition(thyroid$class, p = 0.7, # create index for 70% train set
                                   times = tt, list = FALSE) 

# create two vectors to record AUC for both knn and 
auc.knn <- vector('numeric', tt)
auc.lda <- vector('numeric', tt)
# create a data frame containing different k values to choose from
k.grid <- expand.grid(k=seq(3, 21, by=2))

# use kNN and LDA for classification
for (i in 1:tt){
    train.feature <- thyroid[train.index[,i], -1]   # training features
    train.label <- thyroid$class[train.index[,i]]   # training labels
    test.feature <- thyroid[-train.index[,i], -1]   # test features
    test.label <- thyroid$class[-train.index[,i]]   # test labels
    
    # kNN (tune k based on AUC)
    fitControl <- trainControl(method = 'repeatedcv',   # 5-fold CV
                               number = 5,   
                               repeats = 5,   # repeat 5 times
                               summaryFunction = twoClassSummary, # compute AUC, sensitivity, specificity etc.
                               classProbs = TRUE)  # get the probabilities in prediction
    # training process
    set.seed(5)
    knn.fit <- train(train.feature, train.label, method = 'knn',
                     trControl = fitControl,
                     metric = 'ROC',  # tune k value based on AUC
                     preProcess = c('center', 'scale'),   # define pre-processing of the data
                     tuneGrid = k.grid)   # tuning values
    # test process
    prob.knn <- predict(knn.fit, test.feature, type = 'prob')
    # record AUC
    roc.knn <- roc(predictor = prob.knn$n, response = test.label) 
    auc.knn[i] <- roc.knn$auc
    
    # fit the LDA model
    lda.fit <- train(train.feature, train.label, method = "lda",
                     trControl = trainControl(method = "none"))
    # test process
    prob.lda <- predict(lda.fit, test.feature, type = 'prob')
    # record AUC
    roc.lda <- roc(predictor = prob.lda$n, response = test.label)
    auc.lda[i] <- roc.lda$auc
    
    # extract the ROC curve for first random splitting
    if (i==1){
        roc.knn1 <- roc.knn
        roc.lda1 <- roc.lda
        knn.fit1 <- knn.fit
        lda.fit1 <- lda.fit
        knn.pred1 <- predict(knn.fit, test.feature)
        lda.pred1 <- predict(lda.fit, test.feature)
    }
}

# view the AUC on the test set using classifier kNN and LDA
auc.knn
auc.lda

## Q.ii) ----
# plot the ROC curve for the first random splitting
plot(roc.knn1, main = 'ROC Curve')
plot(roc.lda1, col = 'blue', add = TRUE)  
# add legend
legend("bottomright", legend=c("kNN","LDA"),
       col=c("black","blue"), lty=c(1,1), cex=1, text.font=2)

## Q.iii) ----
# show box plot of AUC values
auc.comp <- data.frame(AUC = c(auc.knn, auc.lda), 
                       Classifier = c(rep('kNN',20), rep('LDA',20)))
ggplot(auc.comp, aes(x=Classifier, y=AUC, fill=Classifier)) +
    geom_boxplot()

## Q.v) ----
# try tune k based on sensitivity ----
# for the first training/test split
train.feature <- thyroid[train.index[,1], -1]   # training features
train.label <- thyroid$class[train.index[,1]]   # training labels
test.feature <- thyroid[-train.index[,1], -1]   # test features
test.label <- thyroid$class[-train.index[,1]]   # test labels
# kNN (tune k based on sensitivity)
fitControl <- trainControl(method = 'repeatedcv',   # 5-fold CV
                           number = 5,   
                           repeats = 5,   # repeat 5 times
                           summaryFunction = twoClassSummary, # compute AUC, sensitivity, specificity etc.
                           classProbs = TRUE)  # get the probabilities in prediction
# training process
set.seed(5)
knn.fit <- train(train.feature, train.label, method = 'knn',
                 trControl = fitControl,
                 metric = 'Sens',  # tune k value based on sensitivity
                 preProcess = c('center', 'scale'),   # define pre-processing of the data
                 tuneGrid = k.grid)   # tuning values
# test process
knn.pred <- predict(knn.fit, test.feature)
confusionMatrix(knn.pred, test.label) 

# get ROC
prob.knn <- predict(knn.fit, test.feature, type = 'prob')
# record AUC
roc.knn <- roc(predictor = prob.knn$n, response = test.label) 
auc.knn2 <- roc.knn$auc
auc.knn2

## Question 2 ----
source('myFDA.R')
# load library and import data
library(MASS)
data(iris)
# obtain training set
X <- iris[51:150,-5]   # training features
y <- iris[51:150,5]   # training labels

# get the linear discriminant from the self-defined function
coef1 <- myFDA(X = X, y = y)
coef1 <- as.vector(coef1)
coef1

# fit the LDA model
lda.model <- lda(Species~., data = iris[51:150,])
# get the linear discriminant using lda() function
coef2 <- lda.model$scaling   # coefficients of linear discriminants
coef2 <- as.vector(coef2)
coef2

# calculate the cosine similarity
cos <- t(coef1)%*%coef2 / (sqrt(t(coef1)%*%coef1) * sqrt(t(coef2)%*%coef2))
cos 
