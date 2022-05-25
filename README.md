# Twitter-Sentiment-Analysis-using-R

## [Understand the topic](#Understand-the-topic) <br/>
Sentiment analysis is commonly known as "opinion mining" or "emotion artificial intelligence". The purpose of this analysis is to determine the mood of an individual that is expressed through text toward someone or to a topic. Specifically, our model will predict the sentiment of every tweet in the test set. If the tweet is a negative sentiment, our model will label it as 0. If the tweet is a positive sentiment, our model will label it as 1.

## [Source Data](#Source-Data)<br/>
There are two data sets in our analysis. The training data contains 3 columns which are ID, SentimentText, and Sentiment, and 29997 observations. Sentiment is the target feature.

In our test data, there are 2 columns which are ID, SentimentText; and 69992 observations.

## [Our analysis methodology](#Our-analysis-methodology) <br/>
We will explore the data in R. For the sake of our practice, we will be mainly using a classification algorithm, specifically Naive Bayes classifier. This is a supervised machine learning algorithm that uses Bayes' theorem and assumes that there is an independent relationship between the features.

## [Text Preprocessing](#Text-Preprocessing) <br/>
One of the crucial stages in the analysis is text preprocessing. We will remove numbers and punctuation, filler words, URL links, white space, any unnecessary symbols such as &amp, < > and transform all the characters to lowercase and then, stem the words. The last step is to split the text documents into words.

These steps are crucial because they allow our model to perform better and produce a higher accuracy score.

## [Model performance evaluation and improvement](#Model-performance-evaluation-and-improvement) <br/>
To increase the accuracy of the model, we will test using Laplace Estimator or decrease the number of frequent words when reducing the data dimensionality.

## [R Packages for this sentiment analysis](#R-Packages-for-this-sentiment-analysis) <br/>
"NLP", "tm", "SnowballC", "wordcloud", "e1071", "gmodels"

