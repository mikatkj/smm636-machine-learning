## 1. import libraries ----
library(shiny)
library(shinythemes)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)
## 2. define app UI ----
ui <- fluidPage(
    # use a theme
    theme = shinytheme("sandstone"),
    # add a title
    titlePanel(""),
    # add a header and brief description
    h2('SMM636 Machine Learning Group 12', align = 'center'),
    h3('Classification of Titanic data: What sorts of people were more likely to survive?', align = 'center'),
    h5('This data set contains information (age, gender, ticket class, fare, # of siblings/spouses and # of parents/children aboard the Titanic) of 887 passengers.', align = 'center'),
    # set Tabset w/ DT, DT (tuning cp), RF
    tabsetPanel(
        type = 'tabs', id = 'tabs',
        tabPanel(
            value = 'tab1', title = 'Decision Tree',
            br(),
            # add a layout for sidebar and main area
            sidebarLayout(
                # inputs
                sidebarPanel(
                    # general: % go to training
                    numericInput(inputId = 'p1', label = '% of data used for training',
                                 value = 70, min = 10, max = 100, step = 10),
                    # extra vertical spacing
                    br(),
                    # DT: threshold for cp
                    sliderInput(inputId = 'cp.input', label = 'The threshold complexity parameter (cp)',
                                value = 0.02, min = 0, max = 0.05, step = 0.002),
                    # DT: select plot error rate against cp/leaves
                    radioButtons(inputId = 'error', label = 'Plot error rates against:',
                                 choices = c('Number of leaves' = 'leaves',
                                             'Complexity parameter' = 'cp'),
                                 inline = TRUE),
                   
                    # output: DT under chosen threshold cp
                    tags$hr(style="border-color: black;"),
                    textOutput(outputId = 'cp'),
                    textOutput(outputId = 'leaves'),
                    textOutput(outputId = 'test.error')
                    ),
                # outputs
                mainPanel(
                    # output: DT under chosen threshold cp
                    plotOutput(outputId = 'dt.cp.plot'),
                    # output: plot error rates
                    fluidRow(column(12, align = 'center',
                                    plotOutput(outputId = 'error', 
                                               width = '70%')))
                    )
            )
        ),
        tabPanel(
            value = 'tab2', title = 'Decision Tree (Tuning cp)',
            br(),
            # add a layout for sidebar and main area
            sidebarLayout(
                # inputs
                sidebarPanel(
                    # general: % go to training
                    numericInput(inputId = 'p2', label = '% of data used for training',
                                 value = 70, min = 10, max = 100, step = 10),
                    # extra vertical spacing
                    br(),
                    # CV: k-fold CV repeats n times
                    sliderInput(inputId = 'k', label = 'K-fold cross-validation',
                                value = 10, min = 1, max = 10, step = 1),
                    sliderInput(inputId = 'repeats', label = 'Repeat times of cross-validation',
                                value = 3, min = 1, max = 10, step = 1),
                    # DT: tune length of cp
                    sliderInput(inputId = 'cp.length', label = 'Length of tuning paramter cp',
                                value = 5, min = 1, max = 10, step = 1),
                    # DT: tune a grid of cp
                    
                    # output: DT using tuned cp
                    tags$hr(style="border-color: black;"),
                    textOutput(outputId = 'cv'),
                    textOutput(outputId = 'cp.optimal'),
                    textOutput(outputId = 'test.error.tune')
                    ),
                # outputs
                mainPanel(
                    # output: DT using tuned cp
                    plotOutput(outputId = 'dt.tune.plot'),
                    # output: plot error rates
                    fluidRow(column(12, align = 'center',
                                    plotOutput(outputId = 'error.tune', 
                                               width = '70%')))
                    )
            )
        ),
        tabPanel(
            value = 'tab3', title = 'Bagging & Random Forest',
            br(),
            # add a layout for sidebar and main area
            sidebarLayout(
                # inputs
                sidebarPanel(
                    # general: % go to training
                    numericInput(inputId = 'p3', label = '% of data used for training',
                                 value = 70, min = 10, max = 100, step = 10),
                    # RF: play with ntree and mrty
                    sliderInput(inputId = 'ntree', label = 'Number of trees to grow',
                                value = 100, min = 100, max = 1000, step = 100),
                    sliderInput(inputId = 'mtry', label = 'Number of predictors randomly selected at each split',
                                value = 3, min = 1, max = 6, step = 1),
                    # RF: variable importance against meanDecreaseAcc/Gini
                    radioButtons(inputId = 'importance', label = 'Plot the importance of features based on measure of:',
                                 choices = c('Accuracy' = 'acc',
                                             'Gini index' = 'gini'),
                                 inline = TRUE),
                    
                    # output: Using Bagging or RF
                    tags$hr(style="border-color: black;"),
                    textOutput(outputId = 'ntree'),
                    textOutput(outputId = 'mtry'),
                    textOutput(outputId = 'oob'),
                    textOutput(outputId = 'test.error.rf')
                ),
                # outputs
                mainPanel(
                    # output: compare Bagging vs RF
                    plotOutput(outputId = 'comp.bag.rf'),
                    # output: variable importance
                    fluidRow(column(12, #align = 'center',
                                    plotOutput(outputId = 'importance.plot', 
                                               width = '80%')))
                )
            )
        ),
        tabPanel(
            value = 'tab4', title = 'Random Forest (Tuning mtry)',
            br(),
            # add a layout for sidebar and main area
            sidebarLayout(
                # inputs
                sidebarPanel(
                    # general: % go to training
                    numericInput(inputId = 'p4', label = '% of data used for training',
                                 value = 70, min = 10, max = 100, step = 10),
                    # CV: k-fold CV repeats n times
                    sliderInput(inputId = 'k2', label = 'K-fold cross-validation',
                                value = 5, min = 1, max = 10, step = 1),
                    sliderInput(inputId = 'repeats2', label = 'Repeat times of cross-validation',
                                value = 3, min = 1, max = 10, step = 1),
                    # RF: tune length of mtry
                    sliderInput(inputId = 'mtry.length', label = 'Length of tuning paramter mtry',
                                value = 4, min = 1, max = 6, step = 1),
                    # RF: tune a grid of mtry
                    
                    # output: RF with tuning mtry
                    tags$hr(style="border-color: black;"),
                    textOutput(outputId = 'cv2'),
                    textOutput(outputId = 'mtry.optimal'),
                    textOutput(outputId = 'test.error.tune2')
                ),
                # outputs
                mainPanel(
                    # output: plot accuracy and variable importance
                    fluidRow(column(6, align = 'center',
                                    plotOutput(outputId = 'acc.mtry', 
                                               width = '100%')),
                             column(6, align = 'center',
                                    plotOutput(outputId = 'importance.mtry', 
                                               width = '100%')))
                )
            )
        )
     )
)
## 3. define server logic ----
server <- function(input, output){
    
    ## use reactive expression to get training/test sets ----
    split <- eventReactive(c(input$p1, input$p2, input$p3, input$p4, input$tabs), {
        # choose % of data going to training
        if(input$tabs == 'tab1'){
            p <- input$p1/100
        }else if(input$tabs == 'tab2'){
            p <- input$p2/100
        }else if(input$tabs == 'tab3'){
            p <- input$p3/100
        }else{
            p <- input$p4/100
            }
        # preprocess data
        titanic <- read.csv('titanic.csv')
        titanic <- titanic[,-3]  # delete 'name' column
        titanic$Survived <- ifelse(titanic$Survived==0, 'Died', 'Survived')   # replace 0/1 with indicators of not survived/survived
        titanic$Survived <- factor(titanic$Survived)   # make the label a factor
        titanic$Pclass <- factor(titanic$Pclass)   # make class variable a factor
        titanic$Sex <- factor(titanic$Sex)   # make sex variable a factor
        titanic$Age <- as.integer(titanic$Age)   # cast Age variable (double) to integers
        # prepare training/test set
        predictor <- c('Pclass','Sex','Age','Siblings.Spouses.Aboard',
                       'Parents.Children.Aboard','Fare')   # choose predictors
        set.seed(345)
        idx <- createDataPartition(titanic$Survived, # randomly split training/test set
                                   p = p, list = FALSE, times = 1)
        train <- titanic[idx,append('Survived',predictor)]
        test <- titanic[-idx,append('Survived',predictor)]
        # return training/test set
        list(train = train, test = test)
    })
    
    
    ## use reactive expression to get DT model ----
    dt.input <- eventReactive(input$cp.input, {
        # build decision trees
        cp.input <- input$cp.input   # chosen cp
        # obtain training set
        train <- split()$train
        # create a DT model
        dt.cp <- rpart(Survived ~., data = train, cp = cp.input)
        # return the model
        list(dt.cp = dt.cp)
    })
    
    
    ## use reactive expression to get DT model after tuning parameters ----
    dt <- eventReactive(c(input$k, input$repeats, input$cp.length), {
        # repeat k-fold CV n times to tune the parameter cp
        k <- input$k   # k-fold
        repeats <- input$repeats   # repeat CV n times
        cp.length <- input$cp.length   # number of cp to be tuned
        control.length <- trainControl(method = 'repeatedcv',   
                                       number = k,
                                       repeats = repeats)
        # obtain training set
        train <- split()$train
        # build the tree
        set.seed(1)
        dt.tune <- train(train[,-1], train$Survived,
                         method = 'rpart',
                         tuneLength = cp.length,
                         trControl = control.length)
        # return DT tuning model
        list(dt.tune = dt.tune)
    })
    
    ## use reactive expression to get RF model ----
    rf <- eventReactive(c(input$ntree, input$mtry), {
        # get ntree and mtry
        mtry <- input$mtry
        p <- 6   # number of all the predictors
        ntree <- input$ntree
        # obtain training/test set
        train <- split()$train
        test <- split()$test
        # build the RF model
        set.seed(103)
        rf <- randomForest(Survived ~., data = train, 
                           mtry = mtry, importance = TRUE, ntree = ntree)
        # build the bagging model
        set.seed(103)
        bag <- randomForest(Survived ~., data = train, 
                            mtry = p, importance = TRUE, ntree = ntree)
        # get OOB error rates
        err.oob.rf <- rf$err.rate[,1]
        err.oob.bag <- bag$err.rate[,1] 
        # get error rates when # of trees=ntree
        err.oob <- err.oob.rf[ntree]
        err.test <- mean(predict(rf, test[,-1]) != test$Survived)
        # get importance of predictors
        impor.acc <- randomForest::importance(rf, type = 1)
        impor.gini <- randomForest::importance(rf, type = 2)

        # return results from bagging/RF models
        list(err.oob.rf = err.oob.rf, err.oob.bag = err.oob.bag,
             err.oob = err.oob, err.test = err.test,
             impor.acc = impor.acc, impor.gini = impor.gini)
    })
    
    ## use reactive expression to get tuning RF model ----
    rf.tune <- eventReactive(c(input$k2, input$repeats2, input$mtry.length), {
        # repeat k-fold CV n times to tune the parameter mtry
        k <- input$k2   # k-fold
        repeats <- input$repeats2   # repeat CV n times
        mtry.length <- input$mtry.length   # number of mtry to be tuned
        control.length <- trainControl(method = 'repeatedcv',   
                                       number = k,
                                       repeats = repeats)
        # obtain training set
        train <- split()$train
        # build the RF model
        set.seed(3)
        rf.tune <- train(Survived ~., data = train,
                         method = 'rf',
                         metric = 'Accuracy',
                         tuneLength = mtry.length,
                         trControl = control.length)
        # return the tuning model
        list(rf.tune = rf.tune)
    })
    
    
    ## OUTPUTS ----
    output$cp <- renderText({
        ## textbox: chosen cp value ----
        paste("Your selected threshold of complexity parameter is: cp =", 
              input$cp.input)
    })
    
    output$leaves <- renderText({
        ## textbox: number of terminal leaves ----
        # obtain the DT model
        dt.cp <- dt.input()$dt.cp
        # counts of cases in leaves
        counts <- data.frame(table(dt.cp$where))   
        # get number of leaves
        paste("The terminal leaves of this decision tree is: size =", 
              nrow(counts))
    })
    
    output$test.error <- renderText({
        ## testbox: corresponding test error rate ----
        # obtain the DT model
        dt.cp <- dt.input()$dt.cp
        # obtain test set
        test <- split()$test
        # calculate test error
        pred <- predict(dt.cp, test[,-1], type = 'class')
        error <- 1 - mean(pred==test$Survived)
        # get corresponding test error rate
        paste("The corresponding test error rate is:",
              round(error,4))
    })
    
    output$dt.cp.plot <- renderPlot({
        ## output plot: DT under chosen threshold cp ----
        cp.input <- input$cp.input   # chosen cp
        # obtain training set
        train <- split()$train
        # create a DT model
        dt.cp <- rpart(Survived ~., data = train, cp = cp.input)
        # visualize the DT model
        rpart.plot(dt.cp, 
                   extra = 104, # show fitted class, probs, percentages
                   box.palette = "RdGn", # color scheme of boxes
                   branch.lty = 3, # dotted branch lines
                   shadow.col = "gray", # shadows under the node boxes
                   nn = TRUE, # display the node numbers
                   roundint =TRUE) # round Age variable
    })
    
    output$error <- renderPlot({
        ## output plot: plot error rate against cp/leaves ----
        button <- input$error
        # obtain training/test set
        train <- split()$train
        test <- split()$test
        # set a range of cp values
        cp.range <- seq(0, 0.5, by=0.005)
        len <- length(cp.range)
        # initialize leaves and error rates vectors
        leaves.range <- c(rep(0, len))
        error.train <- c(rep(0, len))
        error.test <- c(rep(0, len))
        # build decision tree models and predict test error rate
        for (i in 1:len){
            dt.iter <- rpart(Survived ~., data = train, cp = cp.range[i])
            # obtain number of terminal leaves
            leaves.range[i] <- nrow(data.frame(table(dt.iter$where)))
            # training error rate
            pred.train <- predict(dt.iter, train[,-1], type = 'class')
            error.train[i] <- 1 - mean(pred.train==train$Survived)
            # test error rate
            pred.test <- predict(dt.iter, test[,-1], type = 'class')
            error.test[i] <- 1 - mean(pred.test==test$Survived)
        }
        # plot according to chosen x-axis
        if(button=='cp'){
            df <- data.frame(cp = rep(cp.range, 2),
                             set = c(rep('test', len), rep('training', len)),
                             error = c(error.test, error.train)*100)
            p <- ggplot(df, aes(x=cp, y=error, group=set)) +
                geom_line(aes(linetype=set, color=set)) +
                geom_point(aes(color=set)) +
                labs(title='Plot of Error Rate Against Complexity Parameter',
                     x ='Complexity Parameter', y = 'Error Rate (%)')
            p + theme(
                legend.position='right',
                legend.title = element_text(size=14), legend.text = element_text(size=14),
                plot.title = element_text(size=14, face='bold', hjust=0.5),
                axis.title.x = element_text(size=14),
                axis.title.y = element_text(size=14),
                axis.text.x = element_text(size=14),
                axis.text.y = element_text(size=14)
            ) + scale_y_continuous(breaks=c(10,20,30,40), limits=c(10,40))
        }else{
            # get unique numbers of leaves
            leaves <- unique(leaves.range)
            leaves <- rev(leaves)
            len <- length(leaves)
            error.train2 <- c(rep(0, len))
            error.test2 <- c(rep(0, len))
            # get training and test error rates for each number of leaves
            for(i in 1:len){
                index <- min(which(leaves.range==leaves[i]))
                error.test2[i] <- error.test[index]
                error.train2[i] <- error.train[index]
            }
            df <- data.frame(leaves = rep(leaves, 2),
                             set = c(rep('test', len), rep('training', len)),
                             error = c(error.test2, error.train2)*100)
            p <- ggplot(df, aes(x=leaves, y=error, group=set)) +
                geom_line(aes(linetype=set, color=set)) +
                geom_point(aes(color=set)) +
                labs(title='Plot of Error Rate Against Tree Size',
                     x ='Number of Tree Leaves', y = 'Error Rate (%)')
            p + theme(
                legend.position='right',
                legend.title = element_text(size=14), legend.text = element_text(size=14),
                plot.title = element_text(size=14, face='bold', hjust=0.5),
                axis.title.x = element_text(size=14),
                axis.title.y = element_text(size=14),
                axis.text.x = element_text(size=14),
                axis.text.y = element_text(size=14)
            ) + scale_y_continuous(breaks=c(10,20,30,40), limits=c(10,40)) +
                scale_x_continuous(breaks=leaves)
        }
    })
    
    output$cv <- renderText({
        ## textbox: CV parameters ----
        paste("You have selected", input$k, 'fold cross-validation repeated', 
              input$repeats, 'times.')
    })
    
    output$cp.optimal <- renderText({
        ## textbox: optimal cp ----
        # obtain tuned DT model
        dt.tune <- dt()$dt.tune
        # output optimal cp value
        paste('The final cp value used is: cp =', 
              round(dt.tune$finalModel$tuneValue[1,1], 4))
    })
    
    output$test.error.tune <- renderText({
        ## testbox: corresponding test error rate ----
        # obtain tuned DT model
        dt.tune <- dt()$dt.tune
        # obtain test set
        test <- split()$test
        # calculate test error
        pred <- predict(dt.tune, test[,-1], type = 'prob') 
        prob <- pred$Died
        thre.prob <- 0.5   # set threshold=0.5
        pred.dt.tune <- ifelse(prob > thre.prob, 'Died', 'Survived')
        error <- 1 - mean(pred.dt.tune==test$Survived)
        # get corresponding test error rate
        paste("The resulting test error rate is:",
              round(error,4))
    })
    
    
    output$dt.tune.plot <- renderPlot({
        ## output plot: DT using chosen cp ----
        # obtain tuned DT model
        dt.tune <- dt()$dt.tune
        # plot DT
        rpart.plot(dt.tune$finalModel, 
                   extra = 104, # show fitted class, probs, percentages
                   box.palette = "RdGn", # color scheme of boxes
                   branch.lty = 3, # dotted branch lines
                   shadow.col = "gray", # shadows under the node boxes
                   nn = TRUE, # display the node numbers
                   roundint =TRUE)
    })
    
    output$error.tune <- renderPlot({
        ## output plot: training/test error rate against cp ----
        # obtain tuned DT model
        dt.tune <- dt()$dt.tune
        # obtain training/test set
        train <- split()$train
        test <- split()$test
        # get tuning cp and accuracy
        cp.tune <- dt.tune$results[,1]
        accuracy <- dt.tune$results[,2]
        # calculate training error rate
        error.train.tune <- 1 - accuracy
        # initialize test error rate
        len <- length(cp.tune)
        error.test.tune <- c(rep(0, len))
        for (i in 1:len){
            dt.iter <- rpart(Survived ~., data = train, cp = cp.tune[i])
            # test error rate
            pred.test <- predict(dt.iter, test[,-1], type = 'class')
            error.test.tune [i] <- 1 - mean(pred.test==test$Survived)
        }
        # plot error rates against tuning cp values
        df <- data.frame(cp = rep(cp.tune, 2),
                         set = c(rep('test', len), rep('training', len)),
                         error = c(error.test.tune, error.train.tune)*100)
        p <- ggplot(df, aes(x=cp, y=error, group=set)) +
            geom_line(aes(linetype=set, color=set)) +
            geom_point(aes(color=set)) +
            labs(title='Plot of Error Rate Against Tuning CP',
                 x ='Complexity Parameter', y = 'Error Rate (%)')
        p + theme(
            legend.position='right',
            legend.title = element_text(size=14), legend.text = element_text(size=14),
            plot.title = element_text(size=14, face='bold', hjust=0.5),
            axis.title.x = element_text(size=14),
            axis.title.y = element_text(size=14),
            axis.text.x = element_text(size=14),
            axis.text.y = element_text(size=14)
        ) + scale_y_continuous(breaks=c(10,20,30,40), limits=c(10,40))
    })
    
    output$ntree <- renderText({
        ## textbox: chosen ntree ----
        paste("Your selected number of trees to grow is: ntree =", 
              input$ntree)
    })
    
    output$mtry <- renderText({
        ## textbox: chosen mtry ----
        paste("The number of variables tried at each split is: mtry =", 
              input$mtry)
    })
    
    output$oob <- renderText({
        ## textbox: OOB error rate ----
        # obtain the OOB error rate
        err.oob <- rf()$err.oob
        paste("The OOB estimate of error rate is:", round(err.oob, 4))
    })
    
    output$test.error.rf <- renderText({
        ## textbox: test error rate ----
        # obtain the test error rate
        err.test <- rf()$err.test
        paste("The resulting test error rate is:", round(err.test, 4))
    })
    
    output$comp.bag.rf <- renderPlot({
        ## output plot: compare Bagging vs RF ----
        # obtain results from the model
        err.oob.bag <- rf()$err.oob.bag
        err.oob.rf <- rf()$err.oob.rf
        # get ntree and mtry
        mtry <- input$mtry
        p <- 6   # number of all the predictors
        ntree <- input$ntree
        # obtain training/test set
        train <- split()$train
        test <- split()$test
        # compute test error rate (RF)
        gap <- 20  # to reduce computation cost for large # of trees
        err.rate <- vector('numeric', ntree/gap)
        err.test.rf <- vector('numeric', ntree)
        for (i in 1:(ntree/gap)){
            set.seed(103)
            rf.iter <- randomForest(Survived ~., data = train, 
                                    mtry = mtry, importance = TRUE, ntree = (i-1)*gap+1)
            pred.rf <- predict(rf.iter, test[,-1])
            err.rate[i] <- mean(pred.rf != test$Survived)
            for (j in ((i-1)*gap+1):(i*gap)) {err.test.rf[j] <- err.rate[i]}
        }
        for(i in 1:50){
            set.seed(103)
            rf.iter <- randomForest(Survived ~., data = train, 
                                    mtry = mtry, importance = TRUE, ntree = i)
            err.test.rf[i] <- mean(predict(rf.iter, test[,-1]) != test$Survived)
        }
        # compute test error rate (Bagging)
        err.rate <- vector('numeric', ntree/gap)
        err.test.bag <- vector('numeric', ntree)
        for (i in 1:(ntree/gap)){
            set.seed(103)
            bag.iter <- randomForest(Survived ~., data = train, 
                                     mtry = p, importance = TRUE, ntree = (i-1)*gap+1)
            pred.bag <- predict(bag.iter, test[,-1])
            err.rate[i] <- mean(pred.bag != test$Survived)
            for (j in ((i-1)*gap+1):(i*gap)) {err.test.bag[j] <- err.rate[i]}
        }
        for(i in 1:50){
            set.seed(103)
            bag.iter <- randomForest(Survived ~., data = train, 
                                     mtry = p, importance = TRUE, ntree = i)
            err.test.bag[i] <- mean(predict(bag.iter, test[,-1]) != test$Survived)
        }
        # plot line charts comparing oob and test error rates
        len <- ntree
        df <- data.frame(ntree = rep(seq(1:len), 4),
                         Method = c(rep('Test: Bagging', len), rep('Test: RandomForest', len), 
                                    rep('OOB: Bagging', len), rep('OOB: RandomForest', len)),
                         error = c(err.test.bag, err.test.rf,
                                   err.oob.bag, err.oob.rf)*100)
        p <- ggplot(df, aes(x=ntree, y=error, group=Method)) +
            geom_line(aes(linetype=Method, color=Method)) +
            labs(title='Plot of OOB and Test Error Rates Against Number of Trees',
                 x ='Number of Trees', y = 'Error Rate (%)')
        p + theme(
            legend.position='right',
            legend.title = element_text(size=14), legend.text = element_text(size=14),
            plot.title = element_text(size=14, face='bold', hjust=0.5),
            axis.title.x = element_text(size=14),
            axis.title.y = element_text(size=14),
            axis.text.x = element_text(size=14),
            axis.text.y = element_text(size=14)
        ) + scale_y_continuous(breaks=c(15,20,25,30), limits=c(15,30))
    })
    
    output$importance.plot <- renderPlot({
        ## output plot: importance of predictors ----
        impor <- input$importance
        # obtain results from models
        impor.acc <- rf()$impor.acc
        impor.gini <- rf()$impor.gini
        # plot line charts comparing oob and test error rates
        if(impor=='acc'){
            dotchart(sort(impor.acc[,1]), xlim=c(0,100), xlab="Mean Decrease Accuracy",
                     pch = 21, bg = "orange", pt.cex = 1.5,
                     main = 'Importance of Features (with Mean Decrease in Accuracy)')
        }else{
            dotchart(sort(impor.gini[,1]), xlim=c(0,100), xlab="Mean Decrease Gini",
                     pch = 21, bg = "orange", pt.cex = 1.5,
                     main = 'Importance of Features (with Mean Decrease in Node Impurity)')
        }
    })
    
    output$cv2 <- renderText({
        ## textbox: CV parameters ----
        paste("You have selected", input$k2, 'fold cross-validation repeated', 
              input$repeats2, 'times.')
    })
    
    output$mtry.optimal <- renderText({
        ## textbox: optimal mtry ----
        # obtain tuned RF model
        rf.tune <- rf.tune()$rf.tune
        # output optimal mtry value
        paste('The final mtry value used is: mtry =', rf.tune$bestTune[1,1])
    })
    
    output$test.error.tune2 <- renderText({
        ## testbox: corresponding test error rate ----
        # obtain tuned RF model
        rf.tune <- rf.tune()$rf.tune
        # obtain test set
        test <- split()$test
        # calculate test error
        pred <- predict(rf.tune, test[,-1], type = 'prob') 
        prob <- pred$Died
        thre.prob <- 0.5   # set threshold=0.5
        pred.rf.tune <- ifelse(prob > thre.prob, 'Died', 'Survived')
        error <- mean(pred.rf.tune!=test$Survived)
        # get corresponding test error rate
        paste("The resulting test error rate is:", round(error, 4))
    })
    
    output$acc.mtry <- renderPlot({
        ## output plot: plot training accuracy ----
        # obtain tuned RF model
        rf.tune <- rf.tune()$rf.tune
        # plot variable importance
        plot(rf.tune)
    })
    
    output$importance.mtry <- renderPlot({
        ## output plot: plot importance of variables ----
        # obtain tuned RF model
        rf.tune <- rf.tune()$rf.tune
        # plot variable importance
        plot(varImp(rf.tune), top = 6, xlim = c(-10,110))
    })
}
## 4. create shiny app object ----
shinyApp(ui = ui, server = server)