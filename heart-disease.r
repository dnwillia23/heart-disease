##- Creating training and test sets

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")

# Heart Disease dataset:
# https://www.kaggle.com/johnsmith88/heart-disease-dataset
# https://www.kaggle.com/johnsmith88/heart-disease-dataset/downloads/heart-disease-dataset.zip/2

# Reading heart disease dataset from csv file and converting columns to factors
heart <- read.table("./heart.csv", header = TRUE, sep = ",", quote = "\"")
heart = heart %>% mutate(sex = as.factor(sex),
                 cp = as.factor(cp),
                 fbs = as.factor(fbs),
                 restecg = as.factor(restecg),
                 exang = as.factor(exang),
                 slope = as.factor(slope),
                 ca = as.factor(ca),
                 thal = as.factor(thal),
                 target = as.factor(target))

# Validation set will be 20% of heart disease dataset
set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = heart$target, times = 1, p = 0.2, list = FALSE)
train_set <- heart[-test_index,]
test_set <- heart[test_index,]

# Saving full list of stuff above in case RStudio crashes and need to reload
save(train_set,test_set,file = "./trainAndTest.RData")

load("./trainAndTest.RData")

#============= Exploring data ====================
heart %>% group_by(sex) %>% summarize(count = n())
p = ggplot(heart)
p + geom_bar(aes(x = target,fill = sex))
p + geom_boxplot(aes(x = target, y = age))
p + geom_bar(aes(x = target, fill = cp))
p + geom_bar(aes(x = target, fill = thal))
p + geom_bar(aes(x = cp, fill = exang))
p + geom_boxplot(aes(x = target, y = oldpeak))
p + geom_boxplot(aes(x = target, y = chol))

#=============== Ensemble method ======================
models <- c("glm", "lda",  "naive_bayes",  "svmLinear", 
            "knn", "kknn", "gam",
            "rf", "ranger",  "wsrf", 
            "avNNet", "mlp", "monmlp",
            "adaboost", "gbm",
            "svmRadial", "svmRadialCost", "svmRadialSigma")

# Applying ensemble method to model data
# Takes a loooong time to run.  Took about 45 minutes to run on laptop
set.seed(1)
fits <- lapply(models, function(model){ 
  print(model)
  train(target ~ ., method = model, data = train_set)
}) 
names(fits) <- models

# Converting fits to matrix of predicted values with test set
pred <- sapply(fits, function(object)
  predict(object, newdata = test_set))
dim(pred)

# Looking at variable importance of random forest
rf = fits$rf
varImp(rf)

# Getting mean accuracy of each method
acc <- colMeans(pred == test_set$target)
acc
mean(acc)

# Filtering out methods that had mean accuracy of 1
ind1 = which(round(acc,6)!=1)
# ind1 = replicate(length(models),1)
mean(acc[ind1])

# Creating new fits and prediction variables with filtered out methods
new_fits = fits[ind1]
new_pred = pred[,ind1]
new_models = models[ind1]

# Finding ensemble accuracy using majority votes
votes <- rowMeans(new_pred == 1)
y_hat <- ifelse(votes > 0.5, 1, 0)
mean(y_hat == test_set$target)

# Finding which models predict higher than ensemble accuracy
ind <- acc[ind1] > mean(y_hat == test_set$target)
new_models[ind]

# Estimating ensemble accuracy of methods greater than 80%
ind <- (acc_hat >= 0.8)
new_models[ind]
votes <- rowMeans(new_pred[,ind] == 1)
y_hat <- ifelse(votes>=0.5, 1, 0)
mean(y_hat == test_set$target)

# Saving results
save(list = ls(), file = "./results.RData")
