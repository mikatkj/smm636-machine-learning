## Question 1 ----
## 1. Load libraries and import data ----
library(readxl)
library(ggplot2)
library(dplyr)
library(lubridate)
library(plyr)
library(arules)
library(arulesViz)

raw.retail <- read_excel('Online_Retail.xlsx',
                         col_types = c(rep('text',3),'numeric','date','numeric',
                                       'numeric','text'))
retail <- data.frame(raw.retail)

## 2. Pre-process data ----
# check if any missing values in Description
sum(is.na(retail$Description))   # find 1454 missing descriptions
# delete missing values in "Description" column
retail <- retail[complete.cases(retail$Description),]

# have a look at quantity column
summary(retail$Quantity)
sum(retail$Quantity <= 0)   # find remaining 9762 with quantity <= 0
# delete records of quantities <= 0 
retail <- retail[retail$Quantity > 0,]

# have a look at price column
summary(retail$UnitPrice)
sum(retail$UnitPrice <= 0)   # find remaining 589 with price <= 0
# delete records of prices <= 0
retail <- retail[retail$UnitPrice > 0,]

# check unit price & quantity distributions
summary(retail$Quantity)
length(retail$Quantity[retail$Quantity > 10])
length(retail$Quantity[retail$Quantity > 1000])
summary(retail$UnitPrice)
# investigate price outliers
table(retail$Description[retail$UnitPrice > 1000])
# delete undesirable descriptions
retail <- retail[!grepl('Adjust', retail$Description),]
retail <- retail[!grepl('AMAZON', retail$Description),]
retail <- retail[!grepl('POSTAGE', retail$Description),]
retail <- retail[!grepl('Manual', retail$Description),]

## 3. Exploratory analysis ----
# draw boxplot without outliers for UnitPrice
stats <- boxplot.stats(retail$UnitPrice)$stats
df <- data.frame(x='UnitPrice', ymin=stats[1], lower=stats[2], middle=stats[3], 
                 upper=stats[4], ymax=stats[5])
boxplot.price <- ggplot(df, aes(x=x, lower=lower, upper=upper, middle=middle, ymin=ymin, ymax=ymax)) + 
    geom_boxplot(stat='identity') + 
    scale_y_continuous(breaks=stats[1:5], labels=stats[1:5]) + 
    theme(axis.title.x=element_blank(), 
          axis.text.y=element_text(face="bold", size=12),
          axis.text.x=element_text(face="bold", size=12)) 
boxplot.price
# compute np. of outliers
df   # boxplot statistics
length(retail$UnitPrice[retail$UnitPrice > df$ymax])

# draw boxplot without outliers for Quantity
stats <- boxplot.stats(retail$Quantity)$stats
df <- data.frame(x='Quantity', ymin=stats[1], lower=stats[2], middle=stats[3], 
                 upper=stats[4], ymax=stats[5])
boxplot.quantity <- ggplot(df, aes(x=x, lower=lower, upper=upper, middle=middle, ymin=ymin, ymax=ymax)) + 
    geom_boxplot(stat='identity') + 
    scale_y_continuous(breaks=stats[1:5], labels=stats[1:5]) + 
    theme(axis.title.x=element_blank(), 
          axis.text.y=element_text(face="bold", size=12),
          axis.text.x=element_text(face="bold", size=12)) 
boxplot.quantity
# compute np. of outliers
df   # boxplot statistics
length(retail$Quantity[retail$Quantity > df$ymax])

# plot top 20 descriptions by no. of transactions
# title: Top 20 most purchased products
top.description <- retail %>% 
    group_by(Description) %>% 
    dplyr::summarize(count = n()) %>% 
    top_n(20, wt = count) %>%
    arrange(desc(count)) %>% 
    ggplot(aes(x = reorder(Description, count), y = count)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7, width=0.6) +
    theme(axis.title.x=element_blank(), 
          axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + 
    labs(x = "", y = "Number of Transactions")
top.description
   
# plot top 10 countries by no. of transactions (excl. UK)
# title: Number of transactions across top 10 countries (excl. UK)
nrow(subset(retail, retail$Country == 'United Kingdom'))   # 484082
subset1 <- subset(retail, retail$Country != 'United Kingdom')
top.country <- subset1 %>% 
    group_by(Country) %>% 
    dplyr::summarize(count = n()) %>% 
    top_n(10, wt = count) %>%
    arrange(desc(count)) %>% 
    ggplot(aes(x = reorder(Country, count), y = count)) +
    geom_bar(stat = "identity", fill = "darkred", alpha = 0.7, width=0.6) +
    theme(axis.title.x=element_blank(), 
          axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + 
    labs(x = "", y = "Number of Transactions")
top.country

# process `InvoiceDate` 
# add columns that represent transaction days and hours
retail.date <- retail %>% 
    mutate(Date = as.Date(InvoiceDate)) %>% 
    mutate(Time = as.factor(format(InvoiceDate,"%H:%M:%S"))) %>%
    mutate(Month = format_ISO8601(Date, precision = "ym")) %>%
    mutate(Week = wday(Date, week_start = getOption('lubridate.week.start',1))) %>%
    mutate(Hour = hour(hms(Time)))
# check unique elements of months, days in a week and hours in a day
unique(retail.date$Month)
unique(retail.date$Week)
unique(retail.date$Hour)
# visualize no. of transactions by months (from Dec 2010 to Dec 2011)
date.month <- retail.date %>%
    group_by(Month) %>%
    dplyr::summarize(count = n()) %>%
    ggplot(aes(x = Month, y = count)) + 
    geom_bar(stat="identity", fill="#E69F00", alpha=0.6, width=0.5, colour='#E69F00') +
    #geom_line(colour = '#E69F00') +
    geom_text(aes(label=count), vjust=-0.6, size=3.5) +
    theme(axis.title.x=element_blank()) +
    labs(x='', y='Total Number of Transactions') 
date.month
# by days (count the total number of orders throughout the whole week)
date.week <- retail.date %>%
    group_by(Week) %>%
    dplyr::summarize(count = n()) %>%
    ggplot(aes(x = Week, y = count)) + 
    geom_bar(stat="identity", fill="#E69F00", alpha=0.6, width=0.5, colour='#E69F00') +
    #geom_line(colour = '#E69F00') +
    geom_text(aes(label=count), vjust=-0.6, size=3.5) +
    scale_x_continuous(breaks=c(1,2,3,4,5,6,7),
                       labels=c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")) +
    theme(axis.title.x=element_blank()) +
    labs(x='', y='Total Number of Transactions') +
    ylim(0,105000)
date.week
# by hours (count the total number of orders throughout the whole day)
date.day <- retail.date %>%
    group_by(Hour) %>%
    dplyr::summarize(count = n()) %>%
    ggplot(aes(x = Hour, y = count)) + 
    geom_bar(stat="identity", fill="#E69F00", alpha=0.6, width=0.5, colour='#E69F00') +
    #geom_line(colour = '#E69F00') +
    geom_text(aes(label=count), vjust=-0.6, size=3.5) +
    scale_x_continuous(breaks=unique(retail.date$Hour),
                       labels=unique(retail.date$Hour)) +
    theme(axis.title.x=element_blank()) +
    labs(x='', y='Total Number of Transactions') +
    ylim(0,80500)
date.day

## 4. Market Basket Analysis ----
# transform DF to 'transaction' object 
items <- ddply(retail, c('Invoi.eNo'), 
               function(x) paste(x$Description, collapse = ','))
str(items)
# save to csv file
write.csv(items[,2], "items_list_retail.csv", quote = FALSE, row.names = FALSE)
# read the transaction data for association rules analysis
basket <- read.transactions(file="items_list_retail.csv", 
                            rm.duplicates= TRUE,
                            header=TRUE,
                            format="basket",  
                            sep=",", 
                            cols=NULL,
                            quote="")
# summary of transaction data
summary(basket)
inspect(head(basket))  
#itemFrequencyPlot(basket, topN = 10)

# use the Apriori function to find the association rules
# rules 1: support >= 0.5% and confidence >= 70%, involving 2 to 10 items----
rules1 <- apriori(basket, 
                  parameter = list(sup = 0.005, conf = 0.7,  
                                   minlen = 2, maxlen = 10),
                  control = list(verbose=F))  
# summary of the rules
summary(rules1)
# visualize association rules
plot(rules1)
# inspect and plot top 5 lift rules
inspect(sort(rules1, by = 'lift')[1:5])
top1 <- sort(rules1, by = 'lift')[1:5]
plot(top1, method = 'graph', cex=0.7)

# rules 2: support >= 1% and confidence >= 75%, involving at least 2 items----
rules2 <- apriori(basket, 
                  parameter = list(sup = 0.01, conf = 0.75, minlen = 2),
                  control = list(verbose=F))  
# summary of the rules
summary(rules2)
# visualize association rules
plot(rules2)
# inspect and plot top 10 confidence rules
inspect(sort(rules2, by = 'confidence')[1:10])
top2 <- sort(rules2, by = 'confidence')[1:10]
plot(top2, method = 'graph', cex=0.7)

# rules 3: want to know what customers bought before buying PARTY BUNTING ----
rules3 <- apriori(basket, 
                  parameter = list(supp=0.001, conf=0.9, minlen=2, maxlen=4),
                  appearance = list(default = "lhs", rhs = "PARTY BUNTING"),
                  control = list(verbose=F))
# summary of the rules
summary(rules3)
# inspect and plot top 10 lift rules
inspect(sort(rules3, by = 'lift')[1:10])
top3 <- sort(rules3, by = 'lift')[1:10]
plot(top3, method = 'graph', cex=0.7)

# rules 4: want to know what customers would buy if buying JUMBO BAG RED RETROSPOT ----
rules4 <- apriori(basket, 
                  parameter = list(supp=0.001, conf=0.1, minlen=2, maxlen=10),
                  appearance = list(default = "rhs", 
                                    lhs = "JUMBO BAG RED RETROSPOT"),
                  control = list(verbose=F))
# summary of the rules
summary(rules4)
# inspect and plot top 10 confidence rules
inspect(sort(rules4, by = 'confidence')[1:10])
top4 <- sort(rules4, by = 'confidence')[1:10]
plot(top4, method = 'graph', cex=0.7)

# rules 5: want to know what customers would buy if buying SUGAR ----
rules5 <- apriori(basket, 
                  parameter = list(supp=0.01, conf=0.7, minlen=2),
                  appearance = list(default = "rhs", 
                                    lhs = "SUGAR"),
                  control = list(verbose=F))
# summary of the rules
summary(rules5)
# inspect and plot by decreasing confidence rules
inspect(sort(rules5, by = 'confidence'))
top5 <- sort(rules5, by = 'confidence')
plot(top5, method = 'graph', cex=0.7)
