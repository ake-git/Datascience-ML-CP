# Activity Recognition of Weight Lifting Exercises
### Anna Pietruszewska

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants are used. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways what makes 5 classes in the dataset:  
Class A - exactly according to the specification  
Class B - throwing the elbows to the front  
Class C - lifting the dumbbell only halfway  
Class D - lowering the dumbbell only halfway  
Class E - throwing the hips to the front  

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Summary

The project is based on data from the Weight Lifting Exercises (WLE) Dataset. Several sensors were used to collect data about the quality of the exercise execution (class A as a correct execution of the exercise, and five classes B, C, D, E, F indicating mistakes in weight lifting exercise). The goal of the analysis is to build the model that will properly identify the class A to E. Two methods of algorithms were used: decision tree (CART) and random forest, with k-fold cross-validation (k=5). Random forest performed way better than decision tree, achieving 99.4% accuracy and 0.6% out-of-sample error.

## Reading data

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

20 observations to predict the outcome variable (classe) are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

```{r read}
readData <- read.csv("pml-training.csv", header=T)
readNew <- read.csv("pml-testing.csv", header=T)
```

## Data cleaning and preparation

Training dataset consists of 160 variables and 19622 observations. **Classe** is considered as an outcome variable.

```{r data var obs}
dim(readData)
```

We can do some exloration using str() and summary(). We can see that first seven variables are not necessary in the analysis, because they refer to users' names and excercise timestamps. They will be removed from the further analysis.

```{r summary, results="hide"}
str(readData)
summary(readData)
wled <- readData[,-c(1:7)]
```

There are 34 factor variables in the dataset, remaining variables are integer or numeric. For the needs of the project all factor variables except outcome (classe) will be turned into numeric variables.

```{r class, , warning=FALSE}
table(sapply(wled[1,], class))

for (i in 1:ncol(wled)-1) {
    if (class(wled[,i])=="factor") {
        wled[,i] <- as.numeric(levels(wled[,i]))[wled[,i]]
    }
}

table(sapply(wled[1,], class))
```

Summary showed that many variables values are mainly NA. They should not be included in the model because it could bias its accuracy. After this operation 52 predictors left in the dataset.

```{r NA}
missing <- is.na(wled)
delete <- which(colSums(missing) > 19000)
data <- wled[, -delete]
dim(data)
```

Next, variables variability was checked. Predictors with no or small variability have little predicton value so they should not be taken into account while building the model. NearZeroVar function results indicate that all of predictors have enough variability to be kept in the dataset (FALSE nzv value).

```{r nzv, , warning=FALSE, message=FALSE}
library(caret)
nearZeroVar(data,saveMetrics=TRUE)
```

We split our data into a training dataset (70% of cases) and a testing dataset (30% of cases).

```{r split}
set.seed(1224)
inTrain <- createDataPartition(y=data$classe, p=0.7, list=FALSE)

training <- data[inTrain,]
dim(training)

testing <- data[-inTrain,]
dim(testing)
```

Charts below present the frequency of **classe** variable levels in trainng and testing datasets. Distrubutions in both datasets are very similar and they are rather balanced, with some advantage of classe A frequency.

```{r charts}
traincounts <- round(prop.table(table(training$classe)),3)*100
testcounts <- round(prop.table(table(testing$classe)),3)*100

par(mfrow=c(1,2))
b1 <- barplot(traincounts, main="Frequency of Classe (%) - training set", cex.main=1, ylim=c(0,35),col="blue")
text(b1, traincounts, labels=paste(traincounts,"%",sep=""), pos=3, cex=1)
b2 <- barplot(testcounts, main="Frequency of Classe (%) - testing set", cex.main=1, ylim=c(0,35),col="darkblue")
text(b2, testcounts, labels=paste(testcounts,"%",sep=""), pos=3, cex=1)
```

## Prediction algorithms

To build a prediction model, we will try two algorithms: decision tree and random forest, using _caret_ library. We will train each model using **k-fold cross-validation**, with **k=5**. This way each model will be built and validated 5 times with different sub-training and sub-testing datasets, and it will allow to estimate more reliable out-of-sample error.

### Decision tree

The first attempt was building the model with a decision tree.

```{r tree, message=FALSE}
set.seed(222)
tree <- train(classe ~ ., data=training, method="rpart", trControl=trainControl(method = "cv",number = 5))
tree
tree$finalModel
```

However, model accuracy evaluated on testing dataset was only **55.4%**, with out-of-sample error **44.6%**.

```{r tree accuracy}
predT <- predict(tree, newdata=testing)
confusionMatrix(predT,testing$classe)
```

### Random forest

Due to low accuracy of decision tree model, ensemble learning method for classification - random forest, was applied to the data. Deafult number of trees (n=500) was used.

```{r random forest, cache=TRUE, message=FALSE, warning=FALSE}
set.seed(333)
RFfit <- train(classe ~ .,data=training, method="rf", trControl=trainControl(method = "cv",number = 5,allowParallel = TRUE))
RFfit
```

Accuracy verfied on the testing data was **99.4%**, with out-of-sample error **0.6%**. It is much better result than the one obtained with a single decision tree.

```{r rf accurracy}
predRF <- predict(RFfit, newdata=testing)
confusionMatrix(predRF,testing$classe)
```

We can also check, which variables in random forest model have the highest prediction importance.

```{r rf importance}
varImp(RFfit)
```

## Applying the model to the new observations

We will use the random forest model (as more accurate) to predict the classe of each of 20 observations in pml-testing.csv file. All observations were classified correctly (according to *Course Project Prediction Quiz* result 20/20).

```{r new}
predict(RFfit,readNew)
```

## Conclusion

In case of weight lifting exercises data, random forest appeared to be way better predicton algorithm than a single decision tree. Obtained model evaluated on testing dataset had very high accuracy **99.4%**. The most important predictors were: roll belt, pitch forearm and yaw belt.

## References

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

2. Human Activity Recognition - Weight Lifting Exercises Dataset http://groupware.les.inf.puc-rio.br/har