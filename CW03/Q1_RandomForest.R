## Question 1 ----
# load libraries and import data ----
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(ggplot2)
data(GermanCredit)

# check the data ----
str(GermanCredit)   # check variables type and structure
summary(GermanCredit)   # check statistics summary
attributes(GermanCredit$Class)   # check labels
table(GermanCredit$Class)   # check number of instances belonging to each label
# calculate the ratio of imbalance
ratio.imbalance <- table(GermanCredit$Class)[2]/table(GermanCredit$Class)[1]   
ratio.imbalance
# check whether there are variables that have the same values for 2 classes
feature <- GermanCredit[,-10]
nfeature <- ncol(feature)
ll <- vector("numeric", nfeature)
for(ii in 1:nfeature){
    ll[ii] <- length(unique(feature[,ii])) # extract unique elements of each feature and record the number of unique elements
}
idx.uni <- which(ll==1)   # get the index of columns that have exactly the same values
# delete 2 variables that have the same values for both classes
feature[,idx.uni] <- list(NULL)

# random split ----
set.seed(12)   # for reproducibility
# create index for 70% training set
idx.train <- createDataPartition(GermanCredit$Class, 
                                 p = 0.7, times = 1, list = FALSE)
train.feature <- feature[idx.train,]   # training features
train.label <- GermanCredit$Class[idx.train]   # training labels
test.feature <- feature[-idx.train,]   # test features
test.label <- GermanCredit$Class[-idx.train]   # test labels
# combine into training set
train <- cbind(train.feature, Class = train.label)
# test set
test <- cbind(test.feature, Class = test.label)

## Q.1) decision tree ----
# train decision trees using 10-fold cross-validation
set.seed(12)
dt <- rpart(Class ~., data = train, method = 'class',
            control = rpart.control(xval = 10,   # 10-fold cross-validation
                                    minbucket = 2,   # min number of obs in any terminal node
                                    cp = 0.01))   # threshold of complexity parameter
# print a table of fitted models
printcp(dt)
dt$cptable
# plot (relative) cv errors against tree size and cp
plotcp(dt)
# get the cp value which corresponds to min cv error
cp.best <- dt$cptable[which.min(dt$cptable[,'xerror']),'CP']
cp.best
# prune the decision tree using the best cp
dt.prune <- prune(dt, cp = cp.best)
# visualise the pruned tree
dt.plot <- rpart.plot(dt.prune, 
                      extra = 104, # show fitted class, probs, percentages
                      box.palette = "RdGn", # color scheme of boxes
                      branch.lty = 3, # dotted branch lines
                      shadow.col = "gray", # shadows under the node boxes
                      nn = TRUE, # display the node numbers
                      roundint =TRUE) # round int variable
# compute training error
pred.dt <- predict(dt.prune, train.feature, type = 'class')
err.train <- mean(pred.dt!=train.label)
err.train
# compute test error
pred.dt2 <- predict(dt.prune, test.feature, type = 'class')
err.test <- mean(pred.dt2!=test.label)
err.test

## Q.2) random forest ----
# use caret package to train random forest models
fitControl <- trainControl(method = 'repeatedcv',   # set up training control
                           number = 10,   # 10-fold cross-validation
                           repeats = 3)   # cv repeated 3 times
# set up a grid of tuning parameters
grid <- expand.grid(mtry = c(2,10,20,30,40,50,59))   # number of predictors selected
# training process
set.seed(12)
rf <- train(Class ~., data = train, 
            method = 'rf',
            metric = 'Accuracy',   # specify the metric comparing models
            trControl = fitControl,
            tuneGrid = grid)
rf
# plot the cv accuracy against mtry
plot(rf)
# get the chosen model
rf$finalModel
# get the optimal mtry
mtry.best <- rf$bestTune[1,1]
mtry.best
# compute test error
pred.rf <- predict(rf, test.feature)
err.test.rf <- mean(pred.rf!=test.label)
err.test.rf
# visualise the importance of features
plot(varImp(rf), top = mtry.best, xlim = c(-10,110))   # top 10 variables

## Q.3) ROC curves ----
# obtain the predicted probabilities associated with 2 classes from dt and rf models
prob.dt <- predict(dt.prune, test.feature, type = 'prob')
prob.rf <- predict(rf, test.feature, type = 'prob')
# build ROC curves
roc.dt <- roc(predictor = prob.dt[,1], response = test.label)
roc.rf <- roc(predictor = prob.rf[,1], response = test.label)
# get AUC values
auc.dt <- roc.dt$auc
auc.rf <- roc.rf$auc
# plot ROC curves
plot(roc.dt, main = 'ROC Curve')
plot(roc.rf, col = 'blue', add = TRUE)  
# add legend
legend("bottomright", legend=c("Decision Tree","Random Forest"),
       col=c("black","blue"), lty=c(1,1), cex=1, text.font=2)

