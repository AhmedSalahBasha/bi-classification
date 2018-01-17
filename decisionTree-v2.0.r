#Decision Tree Classification - V2.0

#set working directory and import Data
setwd("~/R/work-space")
dataset <- read.csv("bank-full.csv", 
                    header = TRUE, 
                    sep = ";", 
                    stringsAsFactors = TRUE)
## [1]- Exploring the Dataset
#view the structure of data
str(dataset)

#view the summary of data
summary(dataset)

#proportions of target vaiable (y)
table(dataset$y)
prop.table(table(dataset$y))

# generate new variable 'target' depend on 'y' to two levels (1, 0)
dataset$target <- ifelse(dataset$y == "yes", 1, 0)

#generate a numeric correlation matrix among numeric variables
cor(subset(x=dataset, select = c(age, balance, day, duration, campaign, target)))

#generate a correlation scatterplot among numeric variables
# %%%% Becareful ==>  it will take about 3 minutes %%%%
#pairs(subset(x=dataset, select = c(age, balance, day, duration, campaign, target)))

#generate a table to view the relation between the target and another variables
table(dataset$y, dataset$loan)
table(dataset$y, dataset$housing)
table(dataset$y, dataset$contact)
table(dataset$y, dataset$marital)

#remove the target variable 
dataset$target <- NULL
str(dataset)

## [2]- prepare our model
#splitting the dataset into trainingSet and testSet
#install.packages('caret')
library(caret)
set.seed(123)
splitSet <- createDataPartition(y = dataset$y, p = 0.75, list = FALSE)
trainSet <- dataset[splitSet,]
testSet <- dataset[-splitSet,]

#further splitting the testSet into two testSets
splitTest <- createDataPartition(y = testSet$y, p = 0.50, list = FALSE)
testSet_1 <- testSet[splitTest,]
testSet_2 <- testSet[-splitTest,]

#check the rows and proportions of trainSet and testSet
nrow(trainSet)
nrow(testSet)
nrow(testSet_1)
nrow(testSet_2)
prop.table(table(trainSet$y))
prop.table(table(testSet_1$y))
prop.table(table(testSet_2$y))

## [3]- train our model on trainingSet 
#install.packages('C50')
library(C50)
model <- C5.0(trainSet[-17], trainSet$y)
model
summary(model)

## [4]- prediction performance on testSet_1 and testSet_2
test_prediction1 <- predict(model, testSet_1, type = "class")
summary(test_prediction1)
test_prediction2 <- predict(model, testSet_2, type = "class")
summary(test_prediction2)
test_prediction <- predict(model, testSet, type = "class")
summary(test_prediction)

#compare the prediction results with actual y variable
head(cbind(testSet_1, test_prediction1)[testSet_1$marital=="married",], n=10)
head(cbind(testSet_1, test_prediction1)[testSet_1$marital=="single",], n=10)

head(cbind(testSet_2, test_prediction2)[testSet_2$marital=="married",], n=10)
head(cbind(testSet_2, test_prediction2)[testSet_2$marital=="single",], n=10)


#confusion matrix and mmetric
confusionMatrix(test_prediction1, testSet_1$y)
confusionMatrix(test_prediction2, testSet_2$y)
confusionMatrix(test_prediction, testSet$y)


#install.packages('rminer')
library(rminer)

mmetric(testSet_1$y, test_prediction1, c("ACC", "PRECISION", "TPR", "F1"))
mmetric(testSet_2$y, test_prediction2, c("ACC", "PRECISION", "TPR", "F1"))


##--------- Evaluation ------------
library(ROCR)
test_pred_score = predict(model, newdata = testSet, type = 'prob')
head(test_pred_score)

hist(test_pred_score)

pred = prediction(test_pred_score[,2], testSet$y)
eval = performance(pred, "acc")
plot(eval)


##----- ROC Curve -----
roc = performance(pred, "tpr", "fpr")
plot(roc, colorize=T, main="ROC Curve", ylab="Sensitivity", xlab="1-Specifity")


##----- rebuild our model based on specific variables ------
#vaiables: [ age , balance , duration]
#based on this--> #reference: https://arxiv.org/ftp/arxiv/papers/1503/1503.04344.pdf 
new_model <- C5.0(y ~ age+balance+duration, data=trainSet)
new_model
summary(new_model)


#prediction performance on NEW testSet_1 and testSet_2
new_test_prediction1 <- predict(new_model, testSet_1, type = "class")
summary(new_test_prediction1)
new_test_prediction2 <- predict(new_model, testSet_2, type = "class")
summary(new_test_prediction2)
new_test_prediction <- predict(new_model, testSet, type = "class")
summary(new_test_prediction)


#confusion matrix and mmetric of NEW model
confusionMatrix(new_test_prediction1, testSet_1$y)
confusionMatrix(new_test_prediction2, testSet_2$y)
confusionMatrix(new_test_prediction, testSet$y)


##--------- Evaluation of NEW model ------------
new_test_pred_score = predict(new_model, newdata = testSet, type = 'prob')
head(new_test_pred_score)

hist(new_test_pred_score)

new_pred = prediction(new_test_pred_score[,2], testSet$y)
new_eval = performance(new_pred, "acc")
plot(eval)


##----- ROC Curve -----
new_roc = performance(new_pred, "tpr", "fpr")
plot(new_roc, colorize=T, main="ROC Curve", ylab="Sensitivity", xlab="1-Specifity")





























