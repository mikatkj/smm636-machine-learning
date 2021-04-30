## Question 2 ----
## 1. Load libraries and import data ----
library(dplyr)
library(ggplot2)
library(GGally)
library(caret)
library(gbm)
library(keras) 
library(MASS)
library(pROC)

stock.raw <- read.csv('US-stock.csv')

## 2. Pre-processing and data exploration ----
glimpse(stock.raw)   # check variables structure
summary(stock.raw)   # check statistics summary
colnames(stock.raw)   # list column names
# total no. of observations
n.obs <- nrow(stock.raw)
n.obs

# transform Class/Sector to factor type
stock.raw$Class <- as.factor(stock.raw$Class)
stock.raw$Sector <- as.factor(stock.raw$Sector)

# create a data frame of features
feature <- stock.raw[,-c(1,223,224)]   # exclude name/sector/class

# visualize `Class` distribution among sectors ----
table(stock.raw$Class)    # check no. of instances belonging to each label
ratio.imbalance <- table(stock.raw$Class)[2]/table(stock.raw$Class)[1]   
ratio.imbalance   # the ratio of imbalance
table(stock.raw$Sector)   # check labels across sectors
unique(stock.raw$Sector)   # names of sectors
n.sector <- length(unique(stock.raw$Sector))
# compute no. of class0/1 in each sector
stock.class <- stock.raw %>% 
  mutate(Class1 = ifelse(Class==1,1,0)) %>%
  mutate(Class0 = ifelse(Class==0,1,0)) 
n.class1 <- stock.class %>%
  group_by(Sector) %>%
  dplyr::summarize(count = sum(Class1))
n.class0 <- stock.class %>%
  group_by(Sector) %>%
  dplyr::summarize(count = sum(Class0))
n.class1 <- data.frame(n.class1)
n.class0 <- data.frame(n.class0)
df.class <- data.frame(class = as.factor(c(rep(1,n.sector), rep(0,n.sector))),
                       sector = c(as.character(n.class1[,1]), 
                                  as.character(n.class0[,1])),
                       count = c(n.class1[,2], n.class0[,2]))
plot.class <- ggplot(data=df.class, aes(x=sector, y=count, fill=class)) +
  geom_bar(stat="identity", position = "dodge", alpha = 0.7, width=0.6) +
  theme(legend.position = "right") +
  scale_color_manual(values = c("darkblue", "#E69F00")) +
  scale_fill_manual(values = c("darkblue", "#E69F00")) + 
  coord_flip() +
  labs(x = "Sector", y = "Number of Stocks")
plot.class

# handle singular variable ----
# delete 1 variable that have the same values for 2 classes
table(stock.raw$operatingProfitMargin)
feature[,which(colnames(feature)=='operatingProfitMargin')] <- list(NULL)

# handle missing and zero values ----
# calculate % of NAs for each variable
na.perc <- apply(feature, 2, function(x) sum(is.na(x))) / n.obs
na.perc <- sort(na.perc, decreasing = TRUE)
# get the quantile of NA %
quantile(na.perc)
# set around 75% quantile as the threshold
thres.na <- 0.1
# plot NA % as decreasing order (the top 100)
plot.na <- ggplot(data = NULL, aes(x = reorder(names(na.perc)[1:100], desc(na.perc[1:100])), 
                                   y = na.perc[1:100])) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7, width=0.6) +
  geom_hline(yintercept=0.1, linetype="dashed", color="red") +
  scale_y_continuous(breaks=seq(0,1,0.1), labels=seq(0,1,0.1)) +
  theme(axis.title.x=element_blank(), 
        axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + 
  labs(x = "", y = "% of Missing Values") 
plot.na
# delete variables with % of NAs >= threshold
sum.na <- sum(na.perc >= thres.na)   # there are 54 variables
idx.na <- which(colnames(feature) %in% names(na.perc[1:sum.na]))   # get index of those variables
feature[,idx.na] <- list(NULL)

# calculate % of 0 for remaining variables
zero.perc <- apply(feature, 2, function(x) sum(x %in% 0)) / n.obs
zero.perc <- sort(zero.perc, decreasing = TRUE)
# get the quantile of zero %
quantile(zero.perc)
# set around 75% quantile as the threshold
thres.zero <- 0.24
# plot zero % as decreasing order (the top 100)
plot.zero <- ggplot(data = NULL, aes(x = reorder(names(zero.perc)[1:100], desc(zero.perc[1:100])), 
                                   y = zero.perc[1:100])) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7, width=0.6) +
  geom_hline(yintercept=0.24, linetype="dashed", color="red") +
  scale_y_continuous(breaks=seq(0,1,0.1), labels=seq(0,1,0.1)) +
  theme(axis.title.x=element_blank(), 
        axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + 
  labs(x = "", y = "% of Zero Values") 
plot.zero
# delete variables with % of zeros >= threshold
sum.zero <- sum(zero.perc >= thres.zero)   # there are 43 variables
idx.zero <- which(colnames(feature) %in% names(zero.perc[1:sum.zero]))   # get index of those variables
feature[,idx.zero] <- list(NULL)

# calculate remaining feature numbers
n.feature <- ncol(feature)

# handle extreme values ----
# replace zero with NAs
feature[feature == 0] <- NA
# calculate statistics (omit NAs and zeros) and check ymin/ymax
apply(feature, 2, function(x) boxplot.stats(x)$stats)
# boxplot(feature[,1:10])   # check some variables before replacing
# replace outliers with ymin or ymax
for (i in 1:ncol(feature)){
  ymin <- boxplot.stats(feature[,i])$stats[1]
  ymax <- boxplot.stats(feature[,i])$stats[5]
  idx.ymin <- (feature[,i] < ymin) & (!is.na(feature[,i]))
  idx.ymax <- (feature[,i] > ymax) & (!is.na(feature[,i]))
  feature[idx.ymin,i] <- ymin
  feature[idx.ymax,i] <- ymax
}
# boxplot(feature[,1:10])   # check some variables after replacing

# fill the missing and zero values ----
# create a data frame combining features and sector
feature.sector <- cbind(feature, stock.raw$Sector)
colnames(feature.sector)[ncol(feature.sector)] <- 'Sector'
# replace NAs with mean of each sector (note that above we have replaced zeros with NAs)
feature.sector.mean <- feature.sector %>%
  group_by(Sector) %>%
  mutate_at(vars(-c('Sector')), 
            funs(ifelse(is.na(.), mean(., na.rm = TRUE), .)))
# check NAs
sum(is.na(feature.sector.mean))

#	visualise correlation matrix ----
plot.corr <- ggcorr(feature.sector.mean[, -ncol(feature.sector.mean)], 
                    size = 2, hjust = 1, layout.exp = 30)
plot.corr

## 3. Random split ----
set.seed(12)   # for reproducibility
# create index for 70% training set
idx.train <- createDataPartition(stock.raw$Class, 
                                 p = 0.7, times = 1, list = FALSE)
# training features
train.feature <- feature.sector.mean[idx.train,-ncol(feature.sector.mean)] 
# training labels
train.label <- stock.raw$Class[idx.train]  
# test features
test.feature <- feature.sector.mean[-idx.train,-ncol(feature.sector.mean)]   
# test labels
test.label <- stock.raw$Class[-idx.train]   
# combine into training set
train <- cbind(train.feature, Class = train.label)
# test set
test <- cbind(test.feature, Class = test.label)

## (1) random forest ----
# use caret library to build RF (repeat 5-fold CV 1 time to tune mrty)
fitControl <- trainControl(method = 'repeatedcv',  
                           number = 5,  
                           repeats = 3)  
# set up a grid of tuning parameters
grid.rf <- expand.grid(mtry = c(2,11,30,50,70,90,110,123))   # number of predictors selected
# training process
set.seed(12)
stock.rf <- train(Class ~., data = train, 
                  method = 'rf',
                  metric = 'Accuracy',
                  trControl = fitControl,
                  tuneGrid = grid.rf)
stock.rf
# plot the cv accuracy VS mtry
plot.rf <- plot(stock.rf)
plot.rf
# get the optimal mtry
mtry.best <- stock.rf$bestTune[1,1]
mtry.best
# visualize the importance of features
# plot(varImp(stock.rf), top = mtry.best, xlim = c(-10,110)) 
plot.rf.var <- plot(varImp(stock.rf), top = 20, xlim = c(-10,110))
plot.rf.var


## (2) boosting ----
# transform labels into numeric 0-1
train.boost <- train
test.boost <- test
train.boost$Class <- ifelse(train.boost$Class=='0', 0, 1)
test.boost$Class <- ifelse(test.boost$Class=='0', 0, 1)
# use Bernoulli distribution as loss function----
set.seed(12)
stock.boost <- gbm(Class ~., data = train.boost,
                   distribution = 'bernoulli', n.trees = 500,
                   interaction.depth = 2, shrinkage = 0.01)
# check performance using the out-of-bag (OOB) error
ntree.best.boost <- gbm.perf(stock.boost, method = "OOB")
ntree.best.boost
# print relative influence of variable based on chosen no. of trees
summary(stock.boost, n.trees = ntree.best.boost, plotit = FALSE)$var[1:10]
summary(stock.boost, n.trees = ntree.best.boost, plotit = FALSE)$rel.inf[1:10]
# plot relative influence (top 10)
par(mar=c(5,10,2,2))
summary(stock.boost, 
        cBars = 10, n.trees = ntree.best.boost, 
        plotit = TRUE, las = 2)
# dev.off()   # reset plot margins
# use AdaBoost as loss function ----
set.seed(12)
stock.adaboost <- gbm(Class ~., data = train.boost,
                      distribution = 'adaboost', n.trees = 500,
                      interaction.depth = 2, shrinkage = 0.01)
# check performance using the out-of-bag (OOB) error
ntree.best.adaboost <- gbm.perf(stock.adaboost, method = "OOB")
ntree.best.adaboost
# print relative influence of variable based on chosen no. of trees
summary(stock.adaboost, n.trees = ntree.best.adaboost, plotit = FALSE)$var[1:10]
summary(stock.adaboost, n.trees = ntree.best.adaboost, plotit = FALSE)$rel.inf[1:10]
# plot relative influence (top 10)
par(mar=c(5,10,2,2))
summary(stock.adaboost, 
        cBars = 10, n.trees = ntree.best.adaboost, 
        plotit = TRUE, las = 2)
# dev.off()   # reset plot margins


## (3) neural networks ----
# transform labels into numeric 0-1
train.label.nn <- train.label
test.label.nn <- test.label
train.label.nn <- ifelse(train.label.nn=='0', 0, 1)
test.label.nn <- ifelse(test.label.nn=='0', 0, 1)
# transform labels to one-hot vectors
train.label.nn <- to_categorical(train.label.nn, 2)
test.label.nn <- to_categorical(test.label.nn, 2)

# set up keras model 1 ----
# 2 hidden layers
# middle layer of 60 neurons
stock.nn <- keras_model_sequential()
stock.nn %>%
    layer_dense(units = n.feature, activation = 'sigmoid', input_shape = n.feature) %>%   # input layer
    layer_dense(units = 60, activation = 'sigmoid') %>%   # hidden layer
    layer_dense(units = 2, activation = 'softmax')   # output layer
# print summary of keras model
summary(stock.nn)
# compile the model
stock.nn %>% compile(
    loss = 'categorical_crossentropy',   # objective function
    optimizer = optimizer_rmsprop(),   # optimizer
    metrics = c('accuracy')   # training metric
)
# fit it using training data
set.seed(2)
history <- stock.nn %>% fit(
    as.matrix(train.feature), 
    train.label.nn,
    epochs = 500,
    validation_split = 0.3
)
# plot fitted lines of training/validation losses + accuracies
print(history)
plot(history)
# evaluate the model on the test set
stock.nn %>% 
  evaluate(as.matrix(test.feature), test.label.nn)

# set up keras model 2 ----
# 1 hidden layer
# reduce input variables to 60
stock.nn2 <- keras_model_sequential()
stock.nn2 %>%
  layer_dense(units = 60, activation = 'sigmoid', input_shape = n.feature) %>%   # input layer
  layer_dense(units = 2, activation = 'softmax')   # output layer
# print summary of keras model
summary(stock.nn2)
# compile the model
stock.nn2 %>% compile(
  loss = 'categorical_crossentropy',   # objective function
  optimizer = optimizer_rmsprop(),   # optimizer
  metrics = c('accuracy')   # training metric
)
# fit it using training data
set.seed(2)
history2 <- stock.nn2 %>% fit(
  as.matrix(train.feature), 
  train.label.nn,
  epochs = 100,
  validation_split = 0.3
)
# plot fitted lines of training/validation losses + accuracies
print(history2)
plot(history2)
# evaluate the model on the test set
stock.nn2 %>% 
  keras::evaluate(as.matrix(test.feature), test.label.nn)

# set up keras model 3 ----
# 2 hidden layers
# reduce number of epochs to 100
stock.nn3 <- keras_model_sequential()
stock.nn3 %>%
  layer_dense(units = n.feature, activation = 'sigmoid', input_shape = n.feature) %>%   # input layer
  layer_dense(units = 60, activation = 'sigmoid') %>%   # hidden layer
  layer_dense(units = 2, activation = 'softmax')   # output layer
# print summary of keras model
summary(stock.nn3)
# compile the model
stock.nn3 %>% compile(
  loss = 'categorical_crossentropy',   # objective function
  optimizer = optimizer_rmsprop(),   # optimizer
  metrics = c('accuracy')   # training metric
)
# fit it using training data
set.seed(2)
history3 <- stock.nn3 %>% fit(
  as.matrix(train.feature), 
  train.label.nn,
  epochs = 100,
  validation_split = 0.3
)
# plot fitted lines of training/validation losses + accuracies
print(history3)
plot(history3)
# evaluate the model on the test set
stock.nn3 %>% 
  evaluate(as.matrix(test.feature), test.label.nn)

# set up keras model 4 ----
# 1 hidden layer
# reduce input variables to 30
stock.nn4 <- keras_model_sequential()
stock.nn4 %>%
  layer_dense(units = 30, activation = 'sigmoid', input_shape = n.feature) %>%   # input layer
  layer_dense(units = 2, activation = 'softmax')   # output layer
# print summary of keras model
summary(stock.nn4)
# compile the model
stock.nn4 %>% compile(
  loss = 'categorical_crossentropy',   # objective function
  optimizer = optimizer_rmsprop(),   # optimizer
  metrics = c('accuracy')   # training metric
)
# fit it using training data
set.seed(2)
history4 <- stock.nn4 %>% fit(
  as.matrix(train.feature), 
  train.label.nn,
  epochs = 100,
  validation_split = 0.3
)
# plot fitted lines of training/validation losses + accuracies
print(history4)
plot(history4)
# evaluate the model on the test set
stock.nn4 %>% 
  keras::evaluate(as.matrix(test.feature), test.label.nn)


## (4) kNN ----
fitControl <- trainControl(method = 'repeatedcv',
                           number = 10,
                           repeats = 5)
set.seed(5)
stock.knn <- train(train.feature, train.label,
                   method = 'knn',
                   trControl = fitControl,
                   metric = 'Accuracy',
                   preProcess = c('center', 'scale'),
                   tuneLength = 10)
stock.knn
plot.knn <- plot(stock.knn)
plot.knn

## (5) LDA + kNN ----
stock.lda <- lda(train.feature, train.label)
train.feature.proj <- predict(stock.lda, train.feature)$x
test.feature.proj <- predict(stock.lda, test.feature)$x

fitControl <- trainControl(method = 'repeatedcv',
                           number = 10,
                           repeats = 5)
set.seed(5)
stock.knn2 <- train(train.feature.proj, train.label, 
                    method = 'knn',
                    trControl = fitControl,
                    metric = 'Accuracy',
                    preProcess = c('center', 'scale'), 
                    tuneLength = 10)
stock.knn2   # only 1 predictor (projected to 1D)
plot.knn2 <- plot(stock.knn2)
plot.knn2


## 4. Assess model performances ----
# compute test error rates ----
# RF
pred.rf <- predict(stock.rf, test.feature)
testerr.rf <- mean(pred.rf!=test.label)
testerr.rf
# Boosting (Bernoulli)
pred.boost <- predict(stock.boost, test.feature, n.trees = ntree.best.boost, 
                      type = 'response')
pred.boost <- ifelse(pred.boost<0.5, 0, 1)   # set threshold=0.5
testerr.boost <- mean(pred.boost!=test.label)
testerr.boost
# AdaBoost
pred.adaboost <- predict(stock.adaboost, test.feature, n.trees = ntree.best.adaboost, 
                         type = 'response')
pred.adaboost <- ifelse(pred.adaboost<0.5, 0, 1)   # set threshold=0.5
testerr.adaboost <- mean(pred.adaboost!=test.label)
testerr.adaboost
# NN (1 hidden layer)
pred.nn <- stock.nn2 %>% 
  evaluate(as.matrix(test.feature), test.label.nn)
testerr.nn <- 1 - pred.nn[2]
names(testerr.nn) <- 'error'
testerr.nn
# kNN
pred.knn <- predict(stock.knn, test.feature)
testerr.knn <- mean(pred.knn!=test.label)
testerr.knn
# LDA + kNN
pred.knn2 <- predict(stock.knn2, test.feature.proj)   
testerr.knn2 <- mean(pred.knn2!=test.label)
testerr.knn2

# ROC curves ----
# obtain the predicted probabilities associated with 2 classes
prob.rf <- predict(stock.rf, test.feature, type = 'prob')
prob.boost <- predict(stock.boost, test.feature, n.trees = ntree.best.boost,
                      type = 'response')
prob.adaboost <- predict(stock.adaboost, test.feature, n.trees = ntree.best.adaboost, 
                         type = 'response')
prob.nn <- stock.nn2 %>%
  predict_proba(as.matrix(test.feature))
prob.knn <- predict(stock.knn, test.feature, type = 'prob')
prob.knn2 <- predict(stock.knn2, test.feature.proj, type = 'prob')
# build ROC curves
roc.rf <- roc(predictor = prob.rf[,1], response = test.label)
roc.boost <- roc(predictor = prob.boost, response = test.label)
roc.adaboost <- roc(predictor = prob.adaboost, response = test.label)
roc.nn <- roc(predictor = prob.nn[,1], response = test.label)
roc.knn <- roc(predictor = prob.knn[,1], response = test.label)
roc.knn2 <- roc(predictor = prob.knn2[,1], response = test.label)
# get AUC values
auc.rf <- roc.rf$auc
auc.boost <- roc.boost$auc
auc.adaboost <- roc.adaboost$auc
auc.nn <- roc.nn$auc
auc.knn <- roc.knn$auc
auc.knn2 <- roc.knn2$auc
# plot ROC curves
plot(roc.rf, main = 'ROC Curve')
plot(roc.boost, col = 'blue', add = TRUE)  
plot(roc.adaboost, col = 'yellow', lty = 2, add = TRUE)
plot(roc.nn, col = 'purple', lty = 2, add = TRUE)
plot(roc.knn, col = 'red', lty = 1, add = TRUE)
plot(roc.knn2, col = 'green', lty = 3, add = TRUE)
# add legend
legend("bottomright", 
       legend=c("Random Forest","Boost (Binomial)","AdaBoost",
                "Neural (60 units)",'kNN','LDA+kNN'),
       col=c("black","blue",'yellow','purple','red','green'), 
       lty=c(1,1,2,2,1,3), cex=0.5, text.font=1)

# compare test errors and AUC values ----
err <- data.frame(testerr.rf, testerr.boost, testerr.adaboost, testerr.nn, testerr.knn, testerr.knn2)
rownames(err) <- 'Error'
colnames(err) <- c('Random Forest', 'Boosting (Binomial)', 'AdaBoost',
                   'Neural (60 units)', 'kNN', 'LDA+kNN')
auc <- data.frame(auc.rf, auc.boost, auc.adaboost, auc.nn, auc.knn, auc.knn2)
rownames(auc) <- 'AUC'
colnames(auc) <- c('Random Forest', 'Boosting (Binomial)', 'AdaBoost',
                   'Neural (60 units)', 'kNN', 'LDA+kNN')
df.comp <- rbind(round(err,3), round(auc,3))


# 5. Analysis per sector ----
# get subset of financial services instances
feature.fin <- subset(feature.sector.mean, 
                      feature.sector.mean$Sector == 'Financial Services')
stock.fin <- subset(stock.raw, stock.raw$Sector == 'Financial Services')
n.financial <- nrow(stock.fin)
n.financial
ratio.imbalance.fin <- table(stock.fin$Class)[2]/table(stock.fin$Class)[1]  
ratio.imbalance.fin
# random split
set.seed(12) 
idx.train.fin <- createDataPartition(stock.fin$Class, 
                                     p = 0.7, times = 1, list = FALSE)
train.feature.fin <- feature.fin[idx.train.fin,-ncol(feature.fin)] 
train.label.fin <- stock.fin$Class[idx.train.fin]  
test.feature.fin <- feature.fin[-idx.train.fin,-ncol(feature.fin)]   
test.label.fin <- stock.fin$Class[-idx.train.fin]   
train.fin <- cbind(train.feature.fin, Class = train.label.fin)
test.fin <- cbind(test.feature.fin, Class = test.label.fin)

## (1) RF ----
fitControl <- trainControl(method = 'repeatedcv',  
                           number = 5,  
                           repeats = 3)  
# set up a grid of tuning parameters
grid.rf <- expand.grid(mtry = c(2,11,30,50,70,90,110,123))
# training process
set.seed(12)
finan.rf <- train(Class ~., data = train.fin, 
                  method = 'rf',
                  metric = 'Accuracy',
                  trControl = fitControl,
                  tuneGrid = grid.rf)
finan.rf
# plot the cv accuracy VS mtry
plot.rf.fin <- plot(finan.rf)
plot.rf.fin
# get the optimal mtry
mtry.best.fin <- finan.rf$bestTune[1,1]
mtry.best.fin

# (2) boosting ----
train.boost.fin <- train.fin
test.boost.fin <- test.fin
train.boost.fin$Class <- ifelse(train.boost.fin$Class=='0', 0, 1)
test.boost.fin$Class <- ifelse(test.boost.fin$Class=='0', 0, 1)
# training process
set.seed(12)
finan.boost <- gbm(Class ~., data = train.boost.fin,
                   distribution = 'bernoulli', n.trees = 500,
                   interaction.depth = 2, shrinkage = 0.01)
# check performance using the out-of-bag (OOB) error
ntree.best.boost.fin <- gbm.perf(finan.boost, method = "OOB")
ntree.best.boost.fin   # 123

## (3) neural networks ----
# build a NN with 1 hidden layers for classification
# transform labels into numeric 0-1
train.label.nnfin <- train.label.fin
test.label.nnfin <- test.label.fin
train.label.nnfin <- ifelse(train.label.nnfin=='0', 0, 1)
test.label.nnfin <- ifelse(test.label.nnfin=='0', 0, 1)
# transform labels to one-hot vectors
train.label.nnfin <- to_categorical(train.label.nnfin, 2)
test.label.nnfin <- to_categorical(test.label.nnfin, 2)

# set up keras model (1 hidden layers)
finan.nn <- keras_model_sequential()
finan.nn %>%
  layer_dense(units = 60, activation = 'sigmoid', input_shape = n.feature) %>%   # input layer
  layer_dense(units = 2, activation = 'softmax')   # output layer
# print summary of keras model
summary(finan.nn)
# compile the model
finan.nn %>% compile(
  loss = 'categorical_crossentropy',   # objective function
  optimizer = optimizer_rmsprop(),   # optimizer
  metrics = c('accuracy')   # training metric
)
# fit it using training data
set.seed(2)
history.fin <- finan.nn %>% fit(
  as.matrix(train.feature.fin), 
  train.label.nnfin,
  epochs = 50,
  validation_split = 0.3
)
# plot fitted lines of training/validation losses + accuracies
print(history.fin)
plot(history.fin)
# evaluate the model on the test set
finan.nn %>% 
  evaluate(as.matrix(test.feature.fin), test.label.nnfin)

## (4) kNN with SMOTE ----
# transform "0/1" to "class0/class1"
train.smote <- train.fin
test.smote <- test.fin
train.smote$Class <- ifelse(train.smote$Class=='0', 'class0', 'class1')
test.smote$Class <- ifelse(test.smote$Class=='0', 'class0', 'class1')
# rebalance the data
fitControl <- trainControl(method = 'repeatedcv',
                           number = 10,
                           repeats = 5, 
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE,
                           sampling = 'smote')   # to rebalance the data
# training process
set.seed(5)
finan.smote <- train(train.feature.fin, train.smote$Class, 
                     method = 'knn',
                     trControl = fitControl,
                     metric = 'ROC',
                     preProcess = c('center', 'scale'),
                     tuneLength = 10)
finan.smote
plot.smote <- plot(finan.smote)
plot.smote

# compute test errors ----
# RF
pred.rf.fin <- predict(finan.rf, test.feature.fin)
testerr.rf.fin <- mean(pred.rf.fin!=test.label.fin)
testerr.rf.fin
# Boosting (Bernoulli)
pred.boost.fin <- predict(finan.boost, test.feature.fin, 
                          n.trees = ntree.best.boost.fin, 
                          type = 'response')
pred.boost.fin <- ifelse(pred.boost.fin<0.5, 0, 1)   # set threshold=0.5
testerr.boost.fin <- mean(pred.boost.fin!=test.label.fin)
testerr.boost.fin
# NN
pred.nn.fin <- finan.nn %>% 
  evaluate(as.matrix(test.feature.fin), test.label.nnfin)
testerr.nn.fin <- 1 - pred.nn.fin[2]
names(testerr.nn.fin) <- 'error'
testerr.nn.fin
# kNN with SMOTE
pred.smote <- predict(finan.smote, test.feature.fin)
testerr.smote <- mean(pred.smote!=test.smote$Class)
testerr.smote

# ROC curves ----
# obtain the predicted probabilities associated with 2 classes
prob.rf.fin <- predict(finan.rf, test.feature.fin, type = 'prob')
prob.boost.fin <- predict(finan.boost, test.feature.fin, 
                          n.trees = ntree.best.boost.fin, 
                          type = 'response')
prob.nn.fin <- finan.nn %>%
  predict_proba(as.matrix(test.feature.fin))
prob.smote <- predict(finan.smote, test.feature.fin, type = 'prob')

# build ROC curves
roc.rf.fin <- roc(predictor = prob.rf.fin[,1], response = test.fin$Class)
roc.boost.fin <- roc(predictor = prob.boost.fin, response = test.fin$Class)
roc.nn.fin <- roc(predictor = prob.nn.fin[,1], response = test.fin$Class)
roc.smote <- roc(predictor = prob.smote[,1], response = test.fin$Class)

# get AUC values
auc.rf.fin <- roc.rf.fin$auc
auc.boost.fin <- roc.boost.fin$auc
auc.nn.fin <- roc.nn.fin$auc
auc.smote <- roc.smote$auc

# plot ROC curves
plot(roc.rf.fin, main = 'ROC Curve')
plot(roc.boost.fin, col = 'blue', add = TRUE)  
plot(roc.nn.fin, col = 'green', lty = 1, add = TRUE)
plot(roc.smote, col = 'red', lty = 1, add = TRUE)
# add legend
legend("bottomright", 
       legend=c('Random Forest','Gradient Boosting','Neural (60 units)','kNN (SMOTE)'),
       col=c("black","blue",'green','red'), 
       lty=c(1,1,1,1), cex=0.5, text.font=1)

# compare test errors and AUC values ----
err.fin <- data.frame(testerr.rf.fin, testerr.boost.fin, 
                      testerr.nn.fin, testerr.smote)
rownames(err.fin) <- 'Test Error'
colnames(err.fin) <- c('Random Forest', 'Gradient Boosting', 
                       'Neural (60 units)', 'kNN (SMOTE)')
auc.fin <- data.frame(auc.rf.fin, auc.boost.fin, auc.nn.fin, auc.smote)
rownames(auc.fin) <- 'AUC'
colnames(auc.fin) <- c('Random Forest', 'Gradient Boosting', 
                       'Neural (60 units)', 'kNN (SMOTE)')
df.comp.fin <- rbind(round(err.fin,3), round(auc.fin,3))
