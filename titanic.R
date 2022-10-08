library(tidyverse)  #for data manipulation, graphs
library(naivebayes) #building naive bayes model 
library(rpart)      #building and training a decision tree model 
library(randomForest) #building model using random forest 
library(class)      #builidng model using k nearest neighbour
library(ggthemes)
#loading data sets, 2 set train and data provided 
test <- read.csv('....\titanic\titanic\test.csv')
train <- read.csv('....\titanic\titanic\train.csv')


#adding a training column to the data set and combining both sets
train$training <- TRUE
test$training <- FALSE
 
data <- bind_rows(train, test)
str(data)


data %>% 
  filter(is.na(Survived)) %>%
  nrow()
#418 the number of rows in the test data that doesn't have the survival
#status as that's what we want to test


#converting variables listed as integers/characters to categorical variables
data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$Sex <- as.factor(data$Sex)
data$Cabin <- as.factor(data$Cabin)
data$Embarked <- as.factor(data$Embarked)
data$Ticket <- as.factor(data$Ticket)


str(data)
summary(data)

#WHATS IN A NAME
#titles of the passengers on the ship 
data$title <- gsub('(.*, )|(\\..*)', '', data$Name)
table(data$Sex, data$title)

rare_title <- c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major'
                , 'Rev', 'Sir', 'the Countess') #combining rare title
#reassigning some of the titles 
data$title[data$title == 'Mile'] <- 'Miss'
data$title[data$title == 'Ms']   <- 'Miss'
data$title[data$title == 'Mme']  <- 'Mrs'
data$title[data$title %in% rare_title] <- 'Rare_Title'

table(data$Sex, data$title)

#pulling out unique surnames from passenger names 
data$surname <- sapply(data$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])
nlevels(factor(data$surname))
#875 unique passenger surnames 

#Do families swim or sink together?
#creating family size variable based on number of siblings or spouse and number of 
#parents or children
data$familysize <- data$SibSp + data$Parch + 1 # +1 to include the passenger themselves

data$family <- paste(data$surname, data$familysize, sep = '_')

#visualizing the relationship between family size and survival 
ggplot(data[1:891,], aes(x=familysize, fill=factor(Survived))) +
  geom_bar(stat = 'count', position = 'dodge') + 
  scale_x_continuous(breaks = c(1:11)) +
  labs(x = 'Family Size') +
  theme_economist()
#from the graph there was a survivalpenalty to singletons and those with family 
#sizes above 4

#grouping family size intp 3 categories, singletons, small and öarge 
data$famsize_grp[data$familysize == 1] <- 'singleton'
data$famsize_grp[data$familysize < 5 & data$familysize > 1] <- 'small'
data$famsize_grp[data$familysize > 4] <- 'large'

mosaicplot(table(data$famsize_grp, data$Survived), 
           main = 'Survival by Famile Size Category',
           shade = TRUE)
#mosiac plot shows and confirms the penalty of survival amonth singletons and 
#large families nut a benefit to small family



#checking the missing data from the Embarked column
filter(data, Embarked == '')

#both passengers have the same cabin, fare, ticket number and were in 1st class.

#creating a database without the two passengers 
embark_true <- data %>%
                filter(PassengerId != 62 & PassengerId != 830)

#counting the passengers by class and embarked
embark_true %>%
  group_by(Embarked, Pclass) %>%
  count()
#by the count, it seems likely that the two passengers departed from Cherbourg
#or Southampton


#box plot of fare by embarked and passenger class
ggplot(embark_true, aes(Embarked, Fare, fill = Pclass)) +
    geom_boxplot() +
#visualizing the $80 fare of the two missing passengers
  geom_hline(aes(yintercept = 80), linetype = 'dashed')
#ticket prices seem to be similar to that of those who 
#embarked from Cherbourg, thus we impute 'C' for thier port

#assigning C as Embarked and confirming 
data[c(62,830), 'Embarked'] <- 'C'
 data[c(62,830),]

#AGE MISSING DATA 
data %>%
  filter(is.na(Age)) %>%
  count()
#263 missing age inputs
#due to cabin having missing values and ticket number and Survived status can't
#be used to predict age, use the remaining variables to build a regression model 
#to predict the age of the 263 passengers

#using Ordinary Least Square regression method to build the model


#box plot of age data
ggplot(data, aes(y=Age)) + geom_boxplot()
#outliers are mostlier at the higher end of the age spectrum 

boxplot.stats(data$Age)

#use the upper whisker threshold to create a dataframe filtered for ages 
#less than or equal to 66 for training our model 

upper_age<- boxplot.stats(data$Age)$stats[5]
filtered_age <- data$Age <=upper_age
summary(data[filtered_age,]$Age)

#linear regression model 
age_model <- lm(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                data = data[filtered_age,])



age_predictor_columns <- data[is.na(data$Age),
                             c('Pclass', 'Sex', 'SibSp', 'Parch',
                               'Fare', 'Embarked')]

#making age predictions 
age_prediction <- predict(age_model, newdata = age_predictor_columns)

#assign age predictions 
data[is.na(data$Age), 'Age'] <- age_prediction

#check summary stats for age of data
summary(data$Age)
#no more NAs in our data set, but negative ages which is unrealistic 


#saving the minimum age in the data set 
min_age <- data %>%
  filter(Age > 0) %>%
  summarise(min = min(Age))

#replace age vales less than zero with min_age
data[data$Age <= 0, 'Age'] <- min_age
summary(data$Age)
 
#PLOT OF AGE DISTRIBUTIONS
ggplot(data, aes(x = Age)) +
  geom_histogram() +
  theme_economist()

#exploring the relationship bewteen the age and survival 
ggplot(data[1:891,], aes(Age, fill= factor(Survived))) +
  geom_histogram() +
  facet_grid( ~Sex)


#Fare value missing data
data[is.na(data$Fare),]

ggplot(data, aes(y=Fare)) + geom_boxplot()
#with the few outliers and the passenger being in 3rd class, model shouldn't be 
#too much on higher ticket prices 

#finding upper whisker of the plot 
boxplot.stats(data$Fare)$stats

#create dataframe with fare price under upper bound(65) for fare prediction model 
upperfare <- boxplot.stats(data$Fare)$stats[5];upperfare
farefiltered <- data$Fare <=upperfare ; farefiltered
summary(data[farefiltered,]$Fare)

#creating fare model
fare_model <- lm(Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked,
                 data=data[farefiltered,])
summary(fare_model)

#predicting and assigning fare for the passenger with the missing fare and
fare_predictor_columns <- data[is.na(data$Fare), c('Pclass','Sex',
                                                   'Age','SibSp',
                                                   'Parch','Embarked')]

fare_prediction <- predict(fare_model, newdata=fare_predictor_columns)

data[is.na(data$Fare), 'Fare'] <- fare_prediction

data[1044, 'Fare'] <- round(data[1044, 'Fare'],2)
data[1044,]




#FEATURE ENGINERRING 
#view count of missing and non missing cabin by survival status in trained dataset 
train %>%
  filter(Cabin == '') %>%
  group_by(Survived) %>%
  count


#MODEL BUILDING 
survived_equation <- as.formula('Survived ~ Pclass + Sex + SibSp + Parch + 
                                Fare + Embarked + Age')

training_clean <- data %>%
  filter(training == TRUE) %>%
  select(-c(Name, Ticket, Cabin, training))
str(training_clean)
training_clean$Survived <- as.factor(training_clean$Survived)


testing_clean <- data %>%
  filter(training == FALSE) %>%
  select(-c(Survived, Name, Ticket, Cabin, training))
str(testing_clean)

#fix embarked levels to 3 since the NA has been replaced 
training_clean$Embarked <- as.character(training_clean$Embarked)
training_clean$Embarked <- as.factor(training_clean$Embarked)
str(training_clean)

testing_clean$Embarked <- as.character(testing_clean$Embarked)
testing_clean$Embarked <- as.factor(testing_clean$Embarked)
str(testing_clean)


#LOGISTIC REGRESSION MODEL
# 0 = did not survived 1 = survived 
log_reg <- glm(survived_equation, training_clean, family = 'binomial')
summary(reg_log)

#NAIVE BAYES MODEL
nbayes <- naive_bayes(survived_equation, training_clean)
summary(nbayes)

#DECISION TREE MODEL
dectree <- rpart(survived_equation, training_clean,method = 'class')
summary(dectree)

#RANDOM FOREST MODEL
#keep number of trees to grow at 500
#number of variables randomly samples at each split sqrt(p)
#where p is the number of predictor variables, 
round(sqrt(7)) = 3

ranforest <- randomForest(survived_equation, data=training_clean,
                          ntree=500, mtry= 3)
summary(ranforest)


#K NEAREST NEIGHBOUR
#create a dummy variable for each factor level showing either a 1 or 0

#survived values for training set
training_labels <- training_clean[ , 'Survived']

training_knn <- training_clean %>%
  select(Pclass, Sex, SibSp, Parch, Fare, Embarked, Age)

knn_training <- training_knn %>%
  mutate(Male = as.integer(ifelse(Sex == 'male',1, 0)),
         Female = as.integer(ifelse(Sex == 'female', 1, 0)),
         Cherbourg = as.integer(ifelse(Embarked == 'C', 1, 0)),
         Southampton = as.integer(ifelse(Embarked== 'S', 1, 0)),
         Queenstown = as.integer(ifelse(Embarked == 'Q', 1, 0)),
         FirstClass = as.integer(ifelse(Pclass == '1', 1, 0)),
         SecondClass = as.integer(ifelse(Pclass == '2', 1, 0)),
         ThirdClass = as.integer(ifelse(Pclass == '3', 1,0))) %>%
  select(SibSp, Parch, Fare, Age, Male, Female, Cherbourg, Queenstown,
         Southampton, FirstClass, SecondClass, ThirdClass)

head(knn_training)

##survived values for training set
testing_knn <- testing_clean %>%
  select(Pclass, Sex, SibSp, Parch, Fare, Embarked, Age)

knn_testing <- testing_knn %>%
  mutate(Male = as.integer(ifelse(Sex == 'male', 1, 0)),
         Female = as.integer(ifelse(Sex == 'female', 1, 0)),
         Cherbourg = as.integer(ifelse(Embarked == 'C', 1, 0)),
         Queenstown = as.integer(ifelse(Embarked == 'Q', 1, 0)),
         Southampton = as.integer(ifelse(Embarked == 'S', 1, 0)),
         FirstClass = as.integer(ifelse(Pclass == '1', 1, 0)),
         SecondClass = as.integer(ifelse(Pclass == '2', 1, 0)),
         ThirdClass = as.integer(ifelse(Pclass == '3', 1, 0))) %>%
  select(SibSp, Parch, Fare, Age, Male, Female, Cherbourg, Queenstown,
         Southampton, FirstClass, SecondClass, ThirdClass)

head(knn_testing)


#PREDICTIONS
train %>%
  group_by(Survived) %>%
  count() %>%
  mutate(Proportion = n / nrow(train))
# ~62% of passengers in the training datat did not survive


#LOGISTIC REGRESSION
log_reg_pre <- testing_clean %>%
  mutate(prob = predict(log_reg, testing_clean),
         Survived = ifelse(prob >0.5, 1,0))
table(log_reg_pre$Survived)
#logistic regression predicted the death of 288 passengers, 


#NAIVE BAYES
bay_pre <- testing_clean %>%
  mutate(Survived = predict(nbayes, testing_clean))
summary(bay_pre)
table(bayes_solutions$Survived)
#naive bayes predicted the death of 295 passengers

#DECISION TREE
dec_tree_pre <- testing_clean %>%
  mutate(Survived = predict(dectree, testing_clean, type='class'))
summary(dec_tree_pre)
#decision tree predics the death on 274 passengers

#RANDOM FOREST 
ran_for_pre <- testing_clean %>%
  mutate(Survived = predict(ranforest, testing_clean))
summary(ran_for_pre)
#random forest predicts the death of 276 passengers

#K NEAREST NEIGHBOUR 
knn_pre <- testing_clean %>%
  mutate(Survived = knn(train=knn_training, test=knn_testing,
                            cl=training_labels, k=30))
summary(knn_pre)
#knn predicts the death of 304 passengers























