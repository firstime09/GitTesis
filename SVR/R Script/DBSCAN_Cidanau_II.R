library(dbscan)
library(readxl)
library(dplyr)
library(e1071)
library(Boruta)
library(caret)
library(raster)
library(dismo)
setwd("D:/FORESTS2020/GITHUB/Plugin/GitTesis/SVR")
# file =read_excel("D:/FORESTS2020/GITHUB/Plugin/GitTesis/SVR/New_Cidanau_580.xlsx")
file =read_excel("D:/FORESTS2020/GITHUB/Plugin/GitTesis/SVR/Data_1048_Yoga.xlsx")
head(file)
dataall <- file[, c(5,7,8,9,10,11,12)]
# data <- file[, c(5:11)]
data<-dataall[-1]
data<-dataall
head(data)
kNNdistplot(data, k = 5)
abline(h=.05, col = "red", lty=2)
# abline(h=.1, col = "red", lty=2)
res <- dbscan(data, eps = .05, minPts = 5)
# res <- dbscan(data, eps = .1, minPts = 10)
res
pairs(data, col = res$cluster + 1L)
dataall$cluster<-res$cluster
file$cluster<-res$cluster
clean<-dataall %>% filter(cluster > 0)
cleanall<-file %>% filter(cluster > 0)
par(mfrow=c(1,2))
plot(clean$Band_4, clean$frci)
plot(dataall$Band_4, dataall$frci)
par(mfrow=c(1,2))
plot(cleanall$Band_4, cleanall$frci)
plot(file$Band_4, file$frci)


## DBSCAN II
db2 <- cleanall[,c(5,7,8,9,10,11,12)]
head(db2,5)
data<-db2[-1]
head(data)
kNNdistplot(data, k = 5)
abline(h=.015, col = "red", lty=2)
# abline(h=.1, col = "red", lty=2)
res <- dbscan(data, eps = .02, minPts = 15)
# res <- dbscan(data, eps = .1, minPts = 10)
res
pairs(data, col = res$cluster + 1L)
dataall$cluster<-res$cluster
file$cluster<-res$cluster
clean<-dataall %>% filter(cluster > 0)
cleanall<-file %>% filter(cluster > 0)
par(mfrow=c(1,2))
plot(clean$Band_4, clean$frci)
plot(dataall$Band_4, dataall$frci)
par(mfrow=c(1,2))
plot(cleanall$Band_4, cleanall$frci)
plot(file$Band_4, file$frci)


## Feature Selection
svrdata <- cleanall[,c(5,7,8,9,10,11,12)]
svrdata <- data
library(Boruta)
# Decide if a variable is important or not using Boruta
boruta_output <- Boruta(frci ~ ., data=na.omit(svrdata), doTrace=2)  # perform Boruta search
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% ("Confirmed")])  # collect Confirmed and Tentative variables
# boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  # collect Confirmed and Tentative variables
print(boruta_signif)  # significant variables
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  # plot variable importance

# Divide data to training and testing ===============================

set.seed(3033)
intrain <- createDataPartition(y = svrdata$frci, p= 0.7, list = FALSE)
training <- svrdata[intrain,]
testing <- svrdata[-intrain,]

dim(training)
dim(testing)
anyNA(svrdata)
## RMSE
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# svr model ==============================================
# if(require(e1071)){
model <- svm(frci ~ . , training)
predictedY <- predict(model, testing)

# points(data$X, predictedY, col = "red", pch=17)

error <- testing$frci - predictedY  # 
svrPredictionRMSE <- rmse(error)  #  


tuneResult <- tune(svm, frci ~ .,  data = training, 
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9)))
# ) 
print(tuneResult) #

# Draw the first tuning graph 
plot(tuneResult) 

# Draw the second tuning graph
tuneResult <- tune(svm, frci ~ .,  data = training,
                   ranges = list(epsilon = seq(0,0.4,0.1), cost = 2^(2:9))
)
tuneResult_test <- tune(svm, frci ~ .,  data = training, epsilon = 0.1, cost = 1)
# tuneResult <- tune(svm, frci ~ .,  data = training, epsilon = 0.5, cost = 16)

print(tuneResult)
plot(tuneResult)

# plot(data, pch=16)
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, testing) 


# 1. 'Actual' and 'Predicted' data
df <- data.frame(testing$frci, tunedModelY)

# 2. R2 Score components

# 2.1. Average of actual data
avr_y_actual <- mean(df$testing.frci)

# 2.2. Total sum of squares
ss_total <- sum((df$testing.frci - avr_y_actual)^2)

# 2.3. Regression sum of squares
ss_regression <- sum((df$tunedModelY - avr_y_actual)^2)

# 2.4. Residual sum of squares
ss_residuals <- sum((df$testing.frci - df$tunedModelY)^2)

# 3. R2 Score
r2 <- 1 - ss_residuals / ss_total

tunedModel_test <- tuneResult_test$best.model
tunedModelY_test <- predict(tunedModel_test, testing) 


error <- testing$frci - tunedModelY
error_test <- testing$frci - tunedModelY_test


# this value can  be different because the best model is determined by cross-validation over randomly shuffled data 
tunedModelRMSE <- rmse(error)  # 2.219642 
tunedModelRMSE_test <- rmse(error_test)  # 2.219642 
# } 
# acc<-tunedModel.accuracy(tunedModelY, testing$frci)
# accuracy<-confusionMatrix(testing$frci, predict(tunedModel, testing))
# accuracy2<-table(tunedModelY, testing$frci)
# accuracy3<-ftable(predict(tunedModel, testing), testing$frci) 

### next step
cleanall$kelas <-cleanall$Class
number <-cleanall %>%
  group_by(kelas) %>%
  summarize(n())
sample <-cleanall %>%
  group_by(kelas)%>%
  sample_n(min(number$`n()`))
write.csv(sample, file = "Dbscan1000Samp_0.05_5.csv")
par(mfrow=c(1,2))
plot(sample$Band_4, sample$frci)
plot(data$Band_4, data$frci)



# 1. 'Actual' and 'Predicted' data
df <- data.frame(testing$frci, tunedModelY)
avr_y_actual <- mean(df$testing.frci)
ss_total <- sum((df$testing.frci - avr_y_actual)^2)
ss_regression <- sum((df$tunedModelY - avr_y_actual)^2)
ss_residuals <- sum((df$testing.frci - df$tunedModelY)^2)
r2 <- 1 - ss_residuals / ss_total

#### estimation
predictors <- stack(list.files(file.path("D:/FORESTS2020/TRAINING/R/Data/FRCI/TRASH"), pattern='img$', full.names=TRUE ))
cidanau<- predict(predictors, tunedModel_test)
plot(cidanau)
predictors_all <- stack(list.files(file.path("D:/FORESTS2020/TRAINING/R/Data/FRCI/RAW/LANDSAT"), pattern='TIF$', full.names=TRUE ))
cidanau_all<- predict(predictors_all, tunedModel)
plot(cidanau_all)

setwd("D:/FORESTS2020/TRAINING/R/Data/FRCI/Cidanau/Result")
pgR<-writeRaster(cidanau, filename = "FRCI_1025.img")
pgR<-writeRaster(cidanau_all, filename = "FRCI_1025_all.img")

