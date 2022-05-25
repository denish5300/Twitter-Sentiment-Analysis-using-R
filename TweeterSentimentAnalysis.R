#Submission Date:02/27/2022

#Problem Description: to develop a Naïve Bayesian model using the 
#training set to predict the sentiment of every tweet in the test set. 
#This is a binary classification problem, where the goal is to classify 
#each tweet into negative sentiment (a class value 0) or a positive (a class value 0). 
#Inputs:              tweeter_training, tweeter_test
#Output file:         Display values on the Console 
#========================================
#NOTE: Observations and comments are written below the respective code. 

#1.read-in tweeter train and tweeter test data sets.
##################################################
train_raw <- read.csv("tweeter_traning.csv", stringsAsFactors = FALSE)  
test_raw <- read.csv("tweeter_test.csv", stringsAsFactors = FALSE)

#file contains multibyte character: encoding to byte 
train_raw$SentimentText <- iconv(enc2utf8(train_raw$SentimentText),sub="byte")
test_raw$SentimentText <- iconv(enc2utf8(test_raw$SentimentText),sub="byte")

#explore data structure
str(train_raw)
str(test_raw)

#in train data set, there are 29997 records., and 2 features which are ID, Sentiment Text.
#and 1 target feature is Sentiment.
#in test data set, there are 69992 records, and  2 features which are ID and Sentiment Text.

#convert target feature Sentiment into factor in training data set:
train_raw$Sentiment <- factor(train_raw$Sentiment)

#check how many is 0 (negative) and how many is 1 (positive) in training data set
table(train_raw$Sentiment)

#percentage of each case
round(prop.table(table(train_raw$Sentiment)) * 100, 1)

#43.4% are negative sentiments while 56.6% are positive sentiments.
#we do not see the class imbalance issue here. 

#2.data preprocessing: train and test data sets
###############################################
library(NLP)
library(tm)
library(stringr)

#reformat SentimentText from tabular format to corpus format
train_corpus <- VCorpus(VectorSource(train_raw$SentimentText))
test_corpus<- VCorpus(VectorSource(test_raw$SentimentText))

#examine the meta data and char length in tweeter corpus file
inspect(train_corpus[1:2])
inspect(test_corpus[1:2])

#view SentimentText
as.character(train_corpus[[1]])
lapply(train_corpus[1:2], as.character)

as.character(test_corpus[[6]])
lapply(test_corpus[7:9], as.character)


#clean up the text 
##################
#remove URL 
removeURL <- function(x) {gsub('http\\S+\\s*', '', x)}
train_corpus_clean <- tm_map(train_corpus, removeURL)
test_corpus_clean <- tm_map(test_corpus, removeURL)

#double check before and after removing url
as.character(train_corpus[[1]])
as.character(train_corpus_clean[[1]])

as.character(test_corpus[[23]])
as.character(test_corpus_clean[[23]])

#remove &amp, &lt;, &gt;
removeAmp <- function(x) {gsub("&amp", "", x)}
removeLt <- function(x) {gsub ("&lt;", "", x)}
removeGt <- function(x) {gsub ("&gt;", "", x)}

train_corpus_clean <- tm_map(train_corpus_clean, content_transformer(removeAmp))
test_corpus_clean <- tm_map(test_corpus_clean, content_transformer(removeAmp))

train_corpus_clean <- tm_map(train_corpus_clean, content_transformer(removeLt))
test_corpus_clean <- tm_map(test_corpus_clean, content_transformer(removeLt))

train_corpus_clean <- tm_map(train_corpus_clean, content_transformer(removeGt))
test_corpus_clean <- tm_map(test_corpus_clean, content_transformer(removeGt))

as.character(train_corpus[[390]])
as.character(train_corpus_clean[[390]])

as.character(test_corpus[[14]])
as.character(test_corpus_clean[[14]])

#reformat to lowercase for all SentimentText
train_corpus_clean <- tm_map(train_corpus_clean, PlainTextDocument)
test_corpus_clean <- tm_map(test_corpus_clean, PlainTextDocument)

train_corpus_clean <- tm_map(train_corpus_clean, content_transformer(tolower))
test_corpus_clean <- tm_map(test_corpus_clean, content_transformer(tolower))

as.character(train_corpus[[2]]) 
as.character(train_corpus_clean[[2]])

as.character(test_corpus[[14]])
as.character(test_corpus_clean[[14]])

#remove all numbers 
train_corpus_clean <- tm_map(train_corpus_clean, removeNumbers)
test_corpus_clean <- tm_map(test_corpus_clean, removeNumbers)

#double check if all numbers are being removed
as.character(train_corpus[[2]])
as.character(train_corpus_clean[[2]])

as.character(test_corpus[[2]])
as.character(test_corpus_clean[[2]])

#remove all filler words
train_corpus_clean <- tm_map(train_corpus_clean, removeWords, stopwords())
test_corpus_clean <- tm_map(test_corpus_clean, removeWords, stopwords())

#double check before and after removing filler words such as i, we, been,..
as.character(train_corpus[[2]])
as.character(train_corpus_clean[[2]])

as.character(test_corpus[[19]])
as.character(test_corpus_clean[[19]])

#remove punctuation
replacePunctuation <- function(x) {gsub("[[:punct:]]+", " ", x)}
train_corpus_clean <- tm_map(train_corpus_clean, replacePunctuation)
test_corpus_clean <- tm_map(test_corpus_clean, replacePunctuation)

#double check before and after removing the punctuation
as.character(train_corpus[[1]])
as.character(train_corpus_clean[[1]])

as.character(test_corpus[[14]])
as.character(test_corpus_clean[[14]])

#stemming process: extract words to their root form
library(SnowballC)
train_corpus_clean <- tm_map(train_corpus_clean, stemDocument)
test_corpus_clean <- tm_map(test_corpus_clean, stemDocument)

#double check before and after
as.character(train_corpus[[2]])
as.character(train_corpus_clean[[2]])

as.character(test_corpus[[21]])
as.character(test_corpus_clean[[21]])

#remove white space
train_corpus_clean <- tm_map(train_corpus_clean, stripWhitespace)
test_corpus_clean <- tm_map(test_corpus_clean, stripWhitespace)

#double check before and after removing white space
as.character(train_corpus[[435]])
as.character(train_corpus_clean[[435]])

as.character(test_corpus[[17]])
as.character(test_corpus_clean[[17]])

#split text documents into words -> tokenization
train_corpus_clean <- tm_map(train_corpus_clean, PlainTextDocument)
train_dtm <- DocumentTermMatrix(train_corpus_clean)
train_dtm$ncol
train_dtm$nrow
train_dtm$dimnames$Terms[1:3]

test_corpus_clean <- tm_map(test_corpus_clean, PlainTextDocument)
test_dtm <- DocumentTermMatrix(test_corpus_clean)
test_dtm$ncol
test_dtm$nrow
test_dtm$dimnames$Terms[1:3]

#in the training data set, after structuring the data, there are 66100 columns and 69992 rows. 
#in the test data set, after structuring the data, there are 37060 columns and 29997 rows.

#splitting train data into training data (75%) and validation data (25%) 
train_dtm_split <- train_dtm[1:52494, ]
valid_dtm_split <- train_dtm[52495:69992, ]

#save labels for train and validation data
train_labels <- train_raw[1:52494, ]$Sentiment
valid_labels <- train_raw[52495:69992, ]$Sentiment

#check proportion
round(prop.table(table(train_labels)) * 100, 1)
round(prop.table(table(valid_labels)) *100, 1)

#visualize the data --> word clouds
library("wordcloud")
wordcloud(train_corpus_clean, min.freq = 800, random.order = FALSE) 

#words that appear at least 800 times in the data: quot, know, just, love,...

#visualize cloud from spam and ham
negative <- subset(train_raw, Sentiment == 0)
positive <- subset(train_raw, Sentiment == 1)

wordcloud(negative$SentimentText, max.words = 100, 
          colors = brewer.pal(8, "Dark2"), scale = c(3, 0.5))
wordcloud(positive$SentimentText, max.words = 100, 
          colors = brewer.pal(8, "Dark2"), scale = c(3, 0.5))

#negative words that appear max 100 times are just, dont, sorry,..
#positive words that appear max 100 times are good, love, get, will,..

#reduce dimensionality for train dtm and test dtm
#################################################
#find the frequency words
train_dtm_split$ncol
freq_words <- findFreqTerms(train_dtm_split, 100)
freq_words[1:10]

#create DTMs with only the frequent terms (i.e, words appearing at least 100 times)
train_dtm_freq <- train_dtm_split[, freq_words]
valid_dtm_freq<- valid_dtm_split[, freq_words]
test_dtm_freq <- test_dtm[, freq_words]

train_dtm_freq$ncol

#we have significantly reduced the dimension of our train data from 66100 columns to 1003.

#3.contruct the Naive Bayes classifier
#####################################
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
} 

train <- apply(train_dtm_freq, MARGIN = 2, convert_counts) ##training set for modeling
valid <- apply(valid_dtm_freq, MARGIN = 2, convert_counts) ##validation set for modeling
test <- apply(test_dtm_freq, MARGIN = 2, convert_counts) ##testing set for prediction

#train a model on the data
library(e1071)
tweeter_classifier <- naiveBayes(train,train_labels)

#4. predict and evaluate the model performance
##############################################
valid_pred <- predict(tweeter_classifier, valid)

#find agreement btween the two vectors
library("gmodels")
CrossTable(valid_labels, valid_pred,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c("actual", "predicted"))


#improve model performance -> use Laplace estimator
tweeter_classifier2 <- naiveBayes(train, train_labels, laplace = 1)
tweeter_valid_pred2 <- predict(tweeter_classifier2, valid)
CrossTable(valid_labels, tweeter_valid_pred2,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c("actual", "predicted"))

#there is not entirely difference in term of model performance between
#the first and the improved model. 
#therefore, we are going to use the first model for the test data set.
#we can see that in the first model, 4703 + 7986 = 12689
#were correctly classified by the model our of 17498 (~72.51%).


#we use the improved model on the test data to make the prediction
test_pred <- predict(tweeter_classifier, test)

#store the results
test_pred <- data.frame("ID" = test_raw$ID, "Sentiment" = test_pred)
write.csv(test_pred, "test_predictions.csv", row.names = FALSE)
