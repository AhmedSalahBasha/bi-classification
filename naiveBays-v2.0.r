#NaiveBayes Classification - V2.0

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

#encoding the target vaiable (y) as a factor of two levels
dataset$y = factor(dataset$y,
                   levels = c('yes', 'no'),
                   labels = c(1, 0))

#proportions of target vaiable (y)
table(dataset$y)
prop.table(table(dataset$y))



## [2]- prepare our model
#splitting the dataset into trainingSet and testSet
#install.packages('caret')
library(caret)
set.seed(123)
splitSet <- createDataPartition(y = dataset$y, p = 0.75, list = FALSE)
trainSet <- dataset[splitSet,]
testSet <- dataset[-splitSet,]


#check the rows and proportions of trainSet and testSet
nrow(trainSet)
nrow(testSet)

prop.table(table(trainSet$y))
prop.table(table(testSet$y))

## [3]- train our model on trainingSet 
library(e1071)
library(rminer)

#create the naiveBayes classifier model
model <- naiveBayes(y ~ ., data = trainSet)
model

## [4]- making predictions
nb_prediction <- predict(model, testSet, type = "class")

confusionMatrix(nb_prediction, testSet$y)

mmetric(testSet$y, nb_prediction, c("ACC", "PRECISION", "TPR", "F1"))


##--------- Evaluation of model ------------
test_pred_score = predict(model, newdata = testSet, type = 'raw')
head(test_pred_score)

hist(test_pred_score)

pred = prediction(test_pred_score[,1], testSet$y)
eval = performance(pred, "acc")
plot(eval)


##----- ROC Curve -----
roc = performance(pred, "tpr", "fpr")
plot(roc, colorize=T, main="ROC Curve", ylab="Sensitivity", xlab="1-Specifity")
