# Name: Ryan Li Jian Tang
# StudentID: 31902626

rm(list = ls())
WAUS = read.csv("WarmerTomorrow2022.csv", stringsAsFactors = TRUE)
L = as.data.frame(c(1:49))
set.seed(31902626)
L = L[sample(nrow(L), 10, replace = FALSE), ] # Sample 10 locations
WAUS = WAUS[(WAUS$Location %in% L),]
WAUS = WAUS[sample(nrow(WAUS), 2000, replace = FALSE), ] # Sample 2000 rows


# Packages needed
library(tree)
library(e1071)
library(ROCR)
library(rpart)
library(adabag)
library(randomForest)


# Making 3 significant figures
options(digits = 3)

# Q1 Finding proportion of days warmer than previous day
table(WAUS$WarmerTomorrow) # Get the proportion of days warmer tomorrow

# For numerical attributes, get their summary
summary(WAUS[1:23])



# Q2 Pre-processing done
WarmTmrw = ifelse(WAUS$WarmerTomorrow == 1, "yes", "no")  # Pre-process results into yes or no, to make it easier to read
WarmTmrw = as.factor(WarmTmrw)
WAUS = cbind(WAUS, WarmTmrw)
cleansed_WAUS = WAUS

# Replace numerical data's NAs with their median
num_columns = c(5:9,11,14:23)
for(i in num_columns){
  cleansed_WAUS[is.na(cleansed_WAUS[,i]), i] = median(cleansed_WAUS[, i], na.rm = TRUE)
}

# Mode function since R does not have a built in function
getmode = function(v) {
  v = na.omit(v)
  uniqv = unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Replacing Categorical data NAs with their mode
cat_columns = c(10,12:13)
for(i in cat_columns){
  cleansed_WAUS[is.na(cleansed_WAUS[,i]), i] = getmode(cleansed_WAUS[, i])
}

# Omit rows without a result for warmer tomorrow
cleansed_WAUS = na.omit(cleansed_WAUS)


# Q3 Dividing data
set.seed(31902626)
train.row = sample(1:nrow(cleansed_WAUS), 0.7*nrow(cleansed_WAUS))
WAUS.train = cleansed_WAUS[train.row, ]
WAUS.test = cleansed_WAUS[-train.row, ]


# Q4 Building classification models

# Decision Tree
set.seed(31902626)
# decision.train = WAUS.train[sample(nrow(WAUS.train), 100, replace = TRUE),] # Making mutliple test samples to reduce 

# Fitting decision tree model
WAUS.decision = tree(WarmTmrw~.-WarmerTomorrow, data=WAUS.train) 

# Naive Bayes
WAUS.naive = naiveBayes(WarmTmrw~.-WarmerTomorrow, data=WAUS.train)

# Bagging 
WAUS.bagging = bagging(WarmTmrw~.-WarmerTomorrow, data=WAUS.train, mfinal=10)

# Boosting
WAUS.boosting = boosting(WarmTmrw~.-WarmerTomorrow, data=WAUS.train, mfinal=10)

# Random Forest
WAUS.forest = randomForest(WarmTmrw~.-WarmerTomorrow, data=WAUS.train)


# Q5 Testing Models + Confusion Matrices
# Decision Tree
decision.predict = predict(WAUS.decision, WAUS.test, type = "class")
table(predicted = decision.predict, actual = WAUS.test$WarmTmrw)


# Naive Bayes
naive.predict = predict(WAUS.naive, WAUS.test)
table(predicted = naive.predict, actual = WAUS.test$WarmTmrw)

# Bagging
bagging.predict = predict.bagging(WAUS.bagging, newdata=WAUS.test)
table(predicted = bagging.predict$class, actual = WAUS.test$WarmTmrw)
# or use 
bagging.predict$confusion

# Boosting
boosting.predict = predict.boosting(WAUS.boosting, WAUS.test)
table(actual = WAUS.test$WarmTmrw, predicted = boosting.predict$class)

# Random Forest
forest.predict = predict(WAUS.forest, WAUS.test)
table(actual = WAUS.test$WarmTmrw, predicted = forest.predict)

# Q6 Plotting ROC Curve
# Get outputs as cofidence levels
# Decision Tree
roc.decision = predict(WAUS.decision, WAUS.test, type="vector")
decision.pred = prediction(roc.decision[,2], WAUS.test$WarmTmrw)
decision.perf = performance(decision.pred, "tpr", "fpr")

# Naive Bayes
roc.naive = predict(WAUS.naive, WAUS.test, type="raw")
naive.pred = prediction(roc.naive[,2], WAUS.test$WarmTmrw)
naive.perf = performance(naive.pred, "tpr", "fpr")

# Bagging
roc.bagging = predict(WAUS.bagging, WAUS.test, type="vector")
bagging.pred = prediction(roc.bagging$prob[,2], WAUS.test$WarmTmrw)
bagging.pref = performance(bagging.pred, "tpr", "fpr")

# Boosting
roc.boosting = predict(WAUS.boosting, WAUS.test, type="vector")
boosting.pred = prediction(roc.boosting$prob[,2], WAUS.test$WarmTmrw)
boosting.perf = performance(boosting.pred, "tpr", "fpr")

# Forest
roc.forest = predict(WAUS.forest, WAUS.test, type ="prob")
forest.pred = prediction(roc.forest[,2], WAUS.test$WarmTmrw)
forest.perf = performance(forest.pred, "tpr", "fpr")

# Plotting
plot(decision.perf, col = "purple", main="ROC of classifiers")
plot(naive.perf, add=TRUE, col = "blue")
plot(bagging.pref, add=TRUE, col = "red")
plot(boosting.perf, add=TRUE, col = "green")
plot(forest.perf, add=TRUE, col = "orange")
abline(0, 1)

legend("bottomright", legend=c("Decision Tree", "Naive Bayes", "Bagging", "Boosting", "Random Forest"), col = c("purple", "blue", "red", "green", "orange"), lty = 1)


# Getting AUC 
auc.decision = performance(decision.pred, "auc")
auc.naive = performance(naive.pred, "auc")
auc.bagging = performance(bagging.pred, "auc")
auc.boosting = performance(boosting.pred, "auc")
auc.forest = performance(forest.pred, "auc")

# Printing Auc values
auc.decision@y.values[[1]]
auc.naive@y.values[[1]]
auc.bagging@y.values[[1]]
auc.boosting@y.values[[1]]
auc.forest@y.values[[1]]


# Q8 Determining important variables
summary(WAUS.decision)
sort(WAUS.bagging$importance, decreasing=TRUE)
sort(WAUS.boosting$importance, decreasing=TRUE)
sort(WAUS.forest$importance[,1], decreasing = TRUE)

# Question 9
# Filter out 8 important attributes, Humidity3pm,WindDir9am, Pressure9am, Temp3pm, MaxTemp, WindDir3pm, WindGustDir, MinTemp

# Remove unneeded variables
f_WAUS = cleansed_WAUS[,c(5:6,10,12:13,17:18,23,25)]
f_WAUS.train = WAUS.train[,c(5:6,10,12:13,17:18,23,25)]
f_WAUS.test = WAUS.test[,c(5:6,10,12:13,17:18,23,25)]

# Perform Cross validation on decision tree to get minimal tree
WAUS.f_decision = tree(WarmTmrw~., data=f_WAUS.train)
plot(WAUS.f_decision)
text(WAUS.f_decision, pretty = 0)

cvtest = cv.tree(WAUS.f_decision, FUN = prune.misclass)
cvtest # Best size is 6 nodes


WAUS.decision.pruned = prune.misclass(WAUS.f_decision, best = 6)
summary(WAUS.decision.pruned)
plot(WAUS.decision.pruned)
text(WAUS.decision.pruned, pretty = 0)

# Testing Model

f_decision.predict = predict(WAUS.f_decision, f_WAUS.test, type = "class")
table(predicted = f_decision.predict, actual = f_WAUS.test$WarmTmrw)

roc.f_decision = predict(WAUS.f_decision, f_WAUS.test, type="vector")
f_decision.pred = prediction(roc.f_decision[,2], f_WAUS.test$WarmTmrw)
f_decision.perf = performance(f_decision.pred, 'auc')
f_decision.perf@y.values[[1]]



# Q10 Creating best tree classifier
# Getting original data set (that still has NA's) with attributes wanted only
og_WAUS = WAUS[c(5:6,10,12:13,17:18,23,25)]
og_WAUS = na.omit(og_WAUS)
dim(og_WAUS)

set.seed(31902626)
train.row = sample(1:nrow(og_WAUS), 0.7*nrow(og_WAUS))
og_WAUS.train = og_WAUS[train.row, ]
og_WAUS.test = og_WAUS[-train.row, ]


# Building new tree with less variables
set.seed(31902626)
mtry = tuneRF(og_WAUS[1:8], og_WAUS$WarmTmrw, ntreeTry = 1000, stepFactor=1.5, improve = 0.01, trace=TRUE, plot = TRUE)  # Inspired from https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/
print(mtry)

WAUS.f_forest = randomForest(WarmTmrw~., data= og_WAUS.train, ntree=1000, mtry=2)


# Improving upon old random forest trees
set.seed(31902626)
mtry2 = tuneRF(cleansed_WAUS[1:23],cleansed_WAUS$WarmTmrw, ntreeTry = 1000, stepFactor=1.5, improve = 0.01, trace=TRUE, plot = TRUE) 
print(mtry2)

WAUS.i_forest = randomForest(WarmTmrw~.-WarmerTomorrow, data=WAUS.train, ntree=1000, mtry = 6)

# Get Results
f_forest.predict = predict(WAUS.f_forest, og_WAUS.test)
table(observed = og_WAUS.test$WarmTmrw, predicted = f_forest.predict)

i_forest.predict = predict(WAUS.i_forest, WAUS.test)
table(observed = WAUS.test$WarmTmrw, predicted = i_forest.predict)

# Plotting and Comparing ROC curves
roc.f_forest = predict(WAUS.f_forest, og_WAUS.test, type ="prob")
f_forest.pred = prediction(roc.f_forest[,2], og_WAUS.test$WarmTmrw)
f_forest.perf = performance(f_forest.pred, "tpr", "fpr")

roc.i_forest = predict(WAUS.i_forest, WAUS.test, type ="prob")
i_forest.pred = prediction(roc.i_forest[,2], WAUS.test$WarmTmrw)
i_forest.perf = performance(i_forest.pred, "tpr", "fpr")


plot(f_forest.perf, col = "purple", main="ROC of Random Forests")
plot(i_forest.perf, col = "red", add=TRUE)
plot(forest.perf, col = "green", add=TRUE)
abline(0,1)
legend("bottomright", legend=c("New Adjusted Forest","Improved Original Forest", "Original Forest"), col = c("purple", "red","green"), lty = 1)

# Getting AUC
auc.f_forest = performance(f_forest.pred, "auc")
auc.i_forest = performance(i_forest.pred, "auc")

auc.f_forest@y.values[[1]]
auc.i_forest@y.values[[1]]
auc.forest@y.values[[1]]


# Q11 

# Pre-processing

library(neuralnet) # Library needed

# Changing Desired output to a numeric 
cleansed_WAUS$WarmerTomorrow = as.numeric(cleansed_WAUS$WarmerTomorrow)

# Creating indicator columns for categorical values 
recoded = model.matrix(~WindGustDir + WindDir9am + WindDir3pm, data = cleansed_WAUS)

# Combine together with desired attributes, Ignore date and time
combined_WAUS = cbind(cleansed_WAUS[,c(4:9,11,14:24)],recoded)

# Normalise numeric values, to reduce the amount of weight large values have on neural network
preproc = preProcess(combined_WAUS[1:17], method=c("range"))
norm = predict(preproc, combined_WAUS[1:17])

# Change values to normalised values
combined_WAUS[1:17] = norm

# Split data into training + testing
set.seed(31902626)
train.row = sample(1:nrow(combined_WAUS), 0.7*nrow(combined_WAUS))
combined_WAUS.train = combined_WAUS[train.row, ]
combined_WAUS.test = combined_WAUS[-train.row, ]

# Automate formula creation for each column
formula = paste(c(colnames(combined_WAUS[1:17])), collapse="+")
formula2 = paste(c(colnames(combined_WAUS[20:64])), collapse = "+")
formula = paste(c(formula, formula2), collapse = "+")
formula = paste(c("WarmerTomorrow~",formula), collapse = "")

# Create Neural network
WAUS.neural = neuralnet(formula=formula, data=combined_WAUS.train)

# Test Model and get confusion matrix
neural.pred = compute(WAUS.neural, combined_WAUS.test)

neural.pred = as.data.frame(round(neural.pred$net.result, 0))

table(actual = combined_WAUS.test$WarmerTomorrow, predicted = neural.pred$V1)


# Calculate AUC value
detach(package:neuralnet, unload = T) # Due to conflicts with packages
library(ROCR)

neural_predictions = predict(WAUS.neural, combined_WAUS.test, type = "raw")

pred = prediction(neural_predictions, combined_WAUS.test$WarmerTomorrow)
perf = performance(pred, "auc")

perf@y.values[[1]]
