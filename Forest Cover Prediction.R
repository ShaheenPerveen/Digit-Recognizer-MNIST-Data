## the train dataset was loaded
## the dataset has 12 features recorded over 30m * 30m patch, the task is to predict the cover_type 
## of that area

dim(train)
summary(train)

#####################################################################################
#### DATA PREPROCESSING ####
############################

## there is a dummy variable for each soil_type, need to reconvert soil_type dummy to soil type

## getting the index of soil_type dummies
grep("Soil_Type", colnames(train1))

train$Soil_Type <- 0   ## creating column for soil type which will vary from 1 to 40, each number for 
## each soil type, thus soil type essentially becomes a categorical variable

for(i in 1:dim(train)[1]){
  for(j in 1:40{
    train1[i,c("Soil_Type")] <- train1[i,c(57)] + train1[i,c(grep("Soil_Type", colnames(train1))[j])]*j
  }
}

## similarly the wilderness_area feature was also in binary format, need to aggregate the binary va;ues to 
## create one column with different levels

train$Wilderness_Area <- 0   ## similar to soil type creating column for wilderness area which will 
## vary from 1 to 4, each number for each swilderness area, thus soil type 
## essentially becomes a categorical variable

for(i in 1:dim(train)[1]){
  for(j in 1:4{
    train1[i,c("Wilderness_Area")] <- train1[i,c(57)] + train1[i,c(grep("Wilderness_Area", colnames(train1))[j])]*j
  }
}

## converting the columns from integer to factor
train$Soil_Type <- as.factor(train$Soil_Type)
train$Wilderness_Area <- as.factor(train$Wilderness_Area)


# extracting important features i.e. excluding the dummy variables in new train set
train1 <- train[,c("Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology", 
                   "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
                   "Hillshade_9am","Hillshade_Noon", "Hillshade_3pm","Horizontal_Distance_To_Fire_Points",
                   "Wilderness_Area","Soil_Type","Cover_Type")]



#####################################################################################
#### EXPLORATORY ANALYSIS #### 
####    DATA VISUALIZATION   ####
##############################

## loading library ggplot for visualization
library(ggplot2)

parameters <- names(train1)


for(i in 1:dim(train1)[2]){
  p <- ggplot(train1, aes(factor(Cover_Type), train1[,parameters[i]]))
  print(p + geom_boxplot() + ylab(parameters[i]))
  print(i)
}


##### DATA IS CLEAN, OUTLIERS VALUES ARE NOT THERE AND NO MISSONG VALUES FOUND

#####################################################################################
###### ANALYSIS OF CORRELATION BETWEEN FEATURES #######
#######################################################
## to check whether the features are correlated, for continuous variables
## correlogram

require(corrgram)
corrgram.data <- train1[, c("Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology", 
                            "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
                            "Hillshade_9am","Hillshade_Noon", "Hillshade_3pm","Horizontal_Distance_To_Fire_Points")]
## change features of factor type to numeric type for inclusion on correlogram

## generate correlogram
corrgram.vars <- c("Elevation","Aspect", "Slope", "Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
                   "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points")

corrgram(corrgram.data[,corrgram.vars], order=FALSE, lower.panel=NULL,
         upper.panel=panel.pie, 
         text.panel=panel.txt, main="Forest Data")

## although some variables were correlated but they cannot be removed based on domain significane



## to check if the categorical variables are dependant or not

library(MASS)       # load the MASS package 
tbl = table(train$Soil_Type, train$Wilderness_Area) 
chisq.test(tbl) 

## the variables comes out to be independant of each other


## to check the dependance of Categorical variable on continuous variables

for (i in 1:10) {
  fit <- aov(Soil_Type ~ train1[,i], data=train1)
  summary(fit) 
}

for (i in 1:10) {
  fit <- aov(Wilderness_Area ~ train1[,i], data=train1)
  summary(fit) 
}

## all the variables were dependant on each other

### CONCLUSION - ALL THE VARIABLES WILL BE USED FOR MODELLING GIVEN THERE DEPENDANCY AND IMPORTANCE



#####################################################################################
#### MODELLING ####
###################

## CONSIDERING THE NATURE OF DATA, AS IT CONTAINS CATEGORICAL VARIABLES, AND IT 
## random forest

set.seed(1)


# Split data

library(caret)
inTraining <- createDataPartition(train1$Cover_Type, p = .8, list = FALSE)
train.data <- train1[ inTraining,]
test.data <- train1[-inTraining,]

### Search grid
cvCtrl = trainControl(method = "repeatedcv", number = 5, repeats = 5)
newGrid = expand.grid(mtry = c(4,6,8))

### try with trees = 500, 1000, 1500, 2000, 2500

classifierRandomForest1 = train(as.factor(Cover_Type) ~ Elevation + Aspect + Slope + Horizontal_Distance_To_Hydrology + Vertical_Distance_To_Hydrology
                                + Horizontal_Distance_To_Roadways + Hillshade_9am + Hillshade_Noon
                                + Hillshade_3pm + Horizontal_Distance_To_Fire_Points +Wilderness_Area + Soil_Type,
                                data=train.data, trControl = cvCtrl, method = "rf", weights = NULL, tuneGrid = newGrid, n.trees = 500)

classifierRandomForest2 = train(as.factor(Cover_Type) ~ Elevation + Aspect + Slope + Horizontal_Distance_To_Hydrology + Vertical_Distance_To_Hydrology
                                + Horizontal_Distance_To_Roadways + Hillshade_9am + Hillshade_Noon
                                + Hillshade_3pm + Horizontal_Distance_To_Fire_Points +Wilderness_Area + Soil_Type,
                                data=train.data, trControl = cvCtrl, method = "rf", weights = NULL, tuneGrid = newGrid, n.trees = 1000)

classifierRandomForest3 = train(as.factor(Cover_Type) ~ Elevation + Aspect + Slope + Horizontal_Distance_To_Hydrology + Vertical_Distance_To_Hydrology
                                + Horizontal_Distance_To_Roadways + Hillshade_9am + Hillshade_Noon
                                + Hillshade_3pm + Horizontal_Distance_To_Fire_Points +Wilderness_Area + Soil_Type,
                                data=train.data, trControl = cvCtrl, method = "rf", weights = NULL, tuneGrid = newGrid, n.trees = 1500)

classifierRandomForest4 = train(as.factor(Cover_Type) ~ Elevation + Aspect + Slope + Horizontal_Distance_To_Hydrology + Vertical_Distance_To_Hydrology
                                + Horizontal_Distance_To_Roadways + Hillshade_9am + Hillshade_Noon
                                + Hillshade_3pm + Horizontal_Distance_To_Fire_Points +Wilderness_Area + Soil_Type,
                                data=train.data, trControl = cvCtrl, method = "rf", weights = NULL, tuneGrid = newGrid, n.trees = 2000)

classifierRandomForest5 = train(as.factor(Cover_Type) ~ Elevation + Aspect + Slope + Horizontal_Distance_To_Hydrology + Vertical_Distance_To_Hydrology
                                + Horizontal_Distance_To_Roadways + Hillshade_9am + Hillshade_Noon
                                + Hillshade_3pm + Horizontal_Distance_To_Fire_Points +Wilderness_Area + Soil_Type,
                                data=train.data, trControl = cvCtrl, method = "rf", weights = NULL, tuneGrid = newGrid, n.trees = 2500)



## Given these models, we can make statistical 
## statements about their performance differences
resamps <- resamples(list(RF1 = classifierRandomForest1,
                          RF2 = classifierRandomForest2,
                          RF3 = classifierRandomForest3,
                          RF4 = classifierRandomForest4,
                          RF5 = classifierRandomForest5))
resamps

## visualizing the performance of different data set
bwplot(resamps)
xyplot(resamps)

## 
##p-value adjustment: bonferroni 
##Upper diagonal: estimates of the difference
##Lower diagonal: p-value for H0: difference = 0

difValues <- diff(resamps)
difValues
summary(difValues)


### best result from RF4, beyond that its leading to overfitting


### PREDICTION USING MODEL
## EXAMPLE WITH ONE MODEL
pred.rf.test <- predict(classifierRandomForest1, newdata = test.data)
pred.rf.test  <- data.frame(pred.rf.test )

table(pred.rf.test)

confusionMatrix(pred.rf.test$pred.rf.test, test.data$Cover_Type)


### THE MODEL WITH N.TREE = 2000 WITH MTRY = 8 gave the best result among random forest

### ATTEMPTS WITH GBM

gbmGrid <-  expand.grid(n.trees = seq(500,2501,500),
                        interaction.depth = c(4, 6, 8),
                        shrinkage = c(0.1, 0.05))

gbm <- train(as.factor(Cover_Type) ~ Elevation + Aspect + Slope + Horizontal_Distance_To_Hydrology + Vertical_Distance_To_Hydrology
             + Horizontal_Distance_To_Roadways + Hillshade_9am + Hillshade_Noon
             + Hillshade_3pm + Horizontal_Distance_To_Fire_Points +Wilderness_Area + Soil_Type,
             data=train.data, method = "gbm", trControl = cvCtrl, 
             verbose = TRUE, tuneGrid = gbmGrid)


### PREDICTION USING MODEL
## EXAMPLE WITH ONE MODEL
pred.rf.test <- predict(gbm, newdata = test.data)
pred.rf.test  <- data.frame(pred.rf.test )

table(pred.rf.test)

confusionMatrix(pred.rf.test$pred.rf.test, test.data$Cover_Type)

resamps <- resamples(list(GB = gbm,
                          RF = classifierRandomForest4))
resamps

## visualizing the performance of different data set
bwplot(resamps)
xyplot(resamps)


##Upper diagonal: estimates of the difference
##Lower diagonal: p-value for H0: difference = 0

difValues <- diff(resamps)
difValues
summary(difValues)


## best result is given by RF4 i.e. the random forest with 2000 trees

## HENCE WE CONCLUDE THAT RANDOM FOREST IS GIVING BEST RESULT WITH 2000 TREES
