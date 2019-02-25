library(dbscan)
library(readxl)
library(dplyr)
library(e1071)
library(Boruta)
library(caret)
library(raster)
library(dismo)
# setwd("C:/Users/user/Dropbox/FORESTS2020/00AllData/Dataframe Sumatra/Data FRCI Window Area_Malta/")
setwd("D:/00RCode/Result/Data Sumatera")
file =read_excel("FRCI_Line_7.xlsx")
# file =read.csv("FRCI_Line_6.csv")
head(file)
dataall <- file[,-c(3,10)] ## Drop column in dataframe
data<-file[,-c(3,10)] ## Drop column in dataframe
head(data)

number <-data %>%
  group_by(Class) %>%
  summarize(n())
sample <-data%>%
  group_by(Class)%>%
  sample_n(min(number$`n()`))
head(sample)
sample<-sample[-2] ## For remove column Class


head(sample)
lst <- as.data.frame(lapply(sample, function(x) round((x-min(x))/(max(x)-min(x)), 3))) 
head(lst)
dataSample<- lst
head(dataSample)
kNNdistplot(dataSample, k = 5)
abline(h=0.2, col = "red", lty=2)

res <- dbscan(dataSample, eps =0.2 , minPts = 5)
res
pairs(dataSample, col = res$cluster + 1L)
dataSample$cluster<-res$cluster
cleanall<-dataSample %>% filter(cluster > 0)
par(mfrow=c(1,2))
plot(cleanall$Band_4, cleanall$frci)
plot(dataSample$Band_4, dataSample$frci)

setwd('D:/00RCode/Result/Data Sumatera/') #---------------------- After running
write.xlsx(cleanall, file = "FRCI_Line_7_Sumatera_78.13N.xlsx")
write.csv(cleanall, file = "FRCI_Line_7_Sumatera_78.13N.csv")
## Feature Selection
svrdata <- cleanall
svrdata <- cleanall[-8]
head(svrdata)
library(Boruta)
# Decide if a variable is important or not using Boruta
boruta_output <- Boruta(frci ~ ., data=na.omit(svrdata), doTrace=2)  # perform Boruta search
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% ("Confirmed")])  # collect Confirmed and Tentative variables
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

model <- svm(frci ~ . , training)
predictedY <- predict(model, testing)


error <- testing$frci - predictedY  # 
svrPredictionRMSE <- rmse(error)  #  


tuneResult <- tune(svm, frci ~ .,  data = training,
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9)))
print(tuneResult) 
plot(tuneResult)

tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, testing)
error <- testing$frci - tunedModelY
tunedModelRMSE <- rmse(error)  # 2.219642

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


