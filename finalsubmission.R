## Code

### Import libraries

library(caret)
library(randomForest)
library(xgboost)
library(dplyr)

### Load training and testing datasets

training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

### Clean training and test sets
### As the testing set has more NA columns, we remove the NA columns from training as well
training <- training[,8:160]
testing <- testing[,8:160]
training <- training[,colSums(is.na(testing))==0]
testing <- testing[,colSums(is.na(testing))==0]

## Split testing set 80/20 for initial assessment of testing set
set.seed(1000)
inTrain<-createDataPartition(training$classe, p=0.8, list=FALSE)
training.train<-training[inTrain,]
training.test<-training[-inTrain,]

### Find highly correlated variables and eliminate from model-fitting
cor_matrix <- cor(training.train[,-which(names(training.train) == "classe")])
cor_matrix_rm <- cor_matrix                  # Modify correlation matrix
cor_matrix_rm[upper.tri(cor_matrix_rm)] <- 0
diag(cor_matrix_rm) <- 0
cor_matrix_rm

training.classe = training.train$classe
training.train <- training.train[, !apply(cor_matrix_rm, 2, function(x) any(x > 0.8))]
training.train$classe = factor(training.classe)
training.test$classe = factor(training.test$classe)
training.train <- training.train[!is.na(training.train$classe),]

training.test <- select(training.test, colnames(training.train))

### Train different models (using rf, gbm, lda and xgboost)
rfmodel <- randomForest(classe ~ ., data=training.train)
gbmmodel <- train(classe ~ ., method="gbm", data=training.train)
ldamodel <- train(classe ~ ., method="lda", data=training.train)
xgdata <- data.matrix(training.train[-which(names(training.train) == "classe")])
xgmodel <- xgboost(data=xgdata, label = as.integer(training.train$classe), nthread=4, nrounds=300)


### Predict outcomes from all three models except XGBoost using test data
rfprediction <- predict(rfmodel, training.test)
gbmprediction <- predict(gbmmodel, training.test)
ldaprediction <- predict(ldamodel, training.test)

### Show confusion matrix to evaluate 
confusionMatrix(rfprediction, training.test$classe)
confusionMatrix(gbmprediction, training.test$classe)
confusionMatrix(ldaprediction, training.test$classe)

### XGBoost Evaluation (I am not that firm on XGBoost, so this is more of an addon for me to train). It also does not perform as well as other 
xgpredict <- predict(xgmodel, data.matrix(training.test[-which(names(training.train) == "classe")]))
xgpredict <- as.integer(round(xgpredict, digits =0))
xgpredictf <- as.character(xgpredict)
xgpredictf[xgpredictf=="0"] <- "A"
xgpredictf[xgpredictf<="1"] <- "A"
xgpredictf[xgpredictf=="2"] <- "B"
xgpredictf[xgpredictf=="3"] <- "C"
xgpredictf[xgpredictf=="4"] <- "D"
xgpredictf[xgpredictf=="5"] <- "E"
xgpredictf[xgpredictf=="6"] <- "E"
confusionMatrix(factor(xgpredictf), training.test$classe)