#NaiveBayes

# Classification

# Importing the dataset
dataset = read.csv('bank-full.csv', header = TRUE, sep = ";")

#specify which columns we will work on 
#==> the reduct set R={Age, Balance, Duration}
#reference: https://arxiv.org/ftp/arxiv/papers/1503/1503.04344.pdf 
#page 4 - right column
dataset = dataset[c(1,6,12,17)]

# Encoding categorical data - (y) to two levels (1, 0)
dataset$y = factor(dataset$y,
                         levels = c('yes', 'no'),
                         labels = c(1, 0))

# Encoding categorical data - (contact)
# dataset$contact = factor(dataset$contact,
#                    levels = c('unknown', 'telephone', 'cellular'),
#                    labels = c(0, 1, 2))
# dataset$contact = as.numeric(levels(dataset$contact))[dataset$contact]

# Encoding categorical data - (month)
# dataset$month = factor(dataset$month,
#                    levels = c('jan', 'feb', 'mar', 'apr', 'may', 'jun',
#                               'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
#                    labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
# dataset$month = as.numeric(levels(dataset$month))[dataset$month]

# Encoding the target feature as factor
dataset$y = factor(dataset$y, levels = c('1', '0'))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
install.packages('caTools')
library(caTools)
set.seed(123)

#75% of dataset will be trainingSet and 25% will be testSet
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling (for improving the ability to plot it with small scale), I'm not sure!
training_set[-4] = scale(training_set[-4])
test_set[-4] = scale(test_set[-4])

# Fitting Naive Bayes classifier to the Training set
#install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-4],
                        y = training_set$y)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-4])

# Making the Confusion Matrix
cm = table(test_set[, 4], y_pred)

##----------------------
# cm
# y_pred
#     1    0
# 1  359  963
# 0  412 9568
# ===========> 1375 incorrect
##----------------------
