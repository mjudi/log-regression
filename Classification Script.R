# Building a simple 1-gram Logistic Regression classification model for Amazon Food reviews.

# We will start by loading the review data set into the variable products.
# For the purpose of this assignment we will only use reviews were rated as useful by at least 2 users.
if (!require(data.table)) install.packages('data.table',repos="http://cran.us.r-project.org",dependencies = T)
if (!require(dplyr)) install.packages('dplyr',repos="http://cran.us.r-project.org",dependencies = T)
if (!require(tidytext)) install.packages("tidytext",repos="http://cran.us.r-project.org",dependencies = T)
if (!require(h2o)) install.packages("h2o",repos="http://cran.us.r-project.org",dependencies = T)
if (!require(tm)) install.packages("tm",repos="http://cran.us.r-project.org",dependencies = T)

#Load Amazon dataset
products =  fread("Reviews.csv")

#Use helpful reviews
products = products[HelpfulnessNumerator >= 2]

#Remove unnecessary columns
products = products[,.(ProductId,review_summary=Summary,rating = Score,review = Text)]

# Extracting Sentiments:

# We will ignore reviews with a neutral sentiment (rating = 3)
products = products[rating !=3]

# Let’s classify reviews with a rating of 4 or higher as positive (1),
# and reviews with a rating of  2 or lower as negative (-1)
# In R, write a function get_sentiment to assign a sentiment label to every value in the rating column.
# Call this column sentiment.
get_sentiment = function(x){
  if(x < 3){
    return(-1)
  } else if (x >=4){
    return(1)
  } else{
    return(NA)
  }}
get_sentiment = Vectorize(get_sentiment,vectorize.args = "x")

products[,sentiment:=get_sentiment(rating)]

# Cleaning the review text:

# Let’s clean and prepare the text so words with additional white spaces or mixed cases are properly standardized.
#   ● Using the tm package in R, convert the review column to a in memory text corpus (collection of text elements) and assign it to review_corpus 
#   ● Write a function clean_text to do the following
#     ○	Remove punctuations
#     ○	Remove numbers
#     ○	Remove extra white spaces
#     ○	Convert all text to lowercase
#   ●	Apply the clean_text function to review_corpus
review_corpus = VCorpus(VectorSource(products$review))

clean_text= function(X){
  X = tm_map(X,removePunctuation)   
  X = tm_map(X,removeNumbers)   
  X = tm_map(X, stripWhitespace)
  X = tm_map(X, content_transformer(tolower))
  return(X)}

review_corpus = clean_text(review_corpus)

# Build a document term matrix with word count vectors:

# Convert review_corpus into a document term matrix of word count vectors called review_dtm.
# Remove sparse terms from the matrix such that the matrix will be at most 99% sparse.
# Lastly, convert a document term matrix to a datatable review_dt.

#Create a doc-term matrix
review_dtm <- DocumentTermMatrix(review_corpus)   
#Remove sparse terms
review_dtm <- removeSparseTerms(review_dtm, 0.99)
#Convert matrix to data.table
review_dt = as.data.table(cbind(1,as.matrix(review_dtm)))

# Alternatively, heres a complete function to build a document term datatable from a vector of texts
create_doc_term_dt = function(text,max_sparsity = .99){
  
  text_corpus = VCorpus(VectorSource(text))
  print("Converted data to VCorpus")
  
  text_corpus = tm_map(text_corpus,removePunctuation)   
  text_corpus = tm_map(text_corpus,removeNumbers)   
  text_corpus = tm_map(text_corpus, stripWhitespace)
  text_corpus = tm_map(text_corpus, content_transformer(tolower))
  
  print("Formatted VCorpus")
  
  #Build the word count vector for each review
  text_dtm <- DocumentTermMatrix(text_corpus)   
  print("Created Doc-Term Matrix")
  
  text_dtm <- removeSparseTerms(text_dtm, max_sparsity)
  print("Removed Sparse Terms")
  
  text_dt = as.data.table(cbind(1,as.matrix(text_dtm)))
  return(text_dt)
}

review_dt = create_doc_term_dt(text = products$review,max_sparsity = .99)

# Splitting the data into training and test sets:
# Let’s do a train/test split with 80% of the reviews in the training set and 20% of the reviews in the test set. 

#To ensure everyone gets consistent results
set.seed(12345)

#Get all row indices
ID = 1:nrow(review_dt)

#Indices for training set
train = sample(x = ID,size = ceiling(length(ID)*.8),replace = F)

#Indices for test set
test = ID[-train]

# Building a sentiment classifier using logistic regression:
# Refer to the Assignment file for the actual functions used here

###Score Function
score_func = function(w,X){
  return(as.matrix(X)%*%matrix(w,ncol=1))
}

###Sigmoid Function
sigmoid_func = function(w,X){
  score = score_func(w,X)
  sigmoid = 1/(1+exp(-score))
  return(sigmoid)
}

#Cost function
cost_func = function(w,X,y){
  observations = length(y)
  
  predictions = sigmoid_func(w,X)
  
  #log(0) = - infinity so add a slight buffer
  predictions[predictions==1] = .00001  
  
  #Take the error when label=1
  class1_cost = matrix(-y,nrow = 1)%*%matrix(log(predictions),ncol = 1)
  
  #Take the error when label=0
  class2_cost = matrix(-(1-y),nrow = 1)%*%matrix(log(1-predictions),ncol = 1)
  
  cost = class1_cost + class2_cost
  
  cost = cost/observations
  
  return(cost)
}

#Update Weights Function
update_weights_func = function(w,X,y,lr){
  N = nrow(X)
  
  #1 - Get Predictions
  predictions = sigmoid_func(w, X)
  
  gradient = t(as.matrix(X))%*%matrix(predictions - y,ncol = 1)
  
  #3 Take the average cost derivative for each feature
  gradient = gradient/N
  
  #4 - Multiply the gradient by our learning rate
  gradient =gradient*lr
  
  #5 - Subtract from our weights to minimize cost
  w_updated = w - gradient
  
  return(w_updated)
}

#Train Function
train_func = function(w,X,lr,y,iters){
  cost_history = c()
  
  for(i in 1:iters){
    w = as.vector(update_weights_func(w = w,X = X,lr = lr,y=y))
    #Calculate error for auditing purposes
    cost = cost_func(X = X, y = y, w = w)
    cost_history[i]= cost
    print(paste("iter: ",i," cost: ",cost))
  }
  return(list(history = cost_history,w = w))
}

# Making Predictions:
# Run the train_func for 100 iterations with a learning rate of .1 and save the weights to w_opt. 

#Select n random weights uniformly between -1 and 1
w_opt=train_func(
  w = runif(n = ncol(review_dt[train]),min = -1,max = 1),
  X = review_dt[train],lr = .1,
  y = products$sentiment[train],iters = 100)

# Quiz Question 1: How many of the weights are positive?
sum(w_opt[["w"]]>0)

# Calculate the score for the test set using the weights from w_opt[["w"]] and the test data set.
# Store the score in a datatable called test_result with a column for the original sentiment (Sentiment) and test indices (ID).
test_results = data.table(ID = test,Sentiment = products$sentiment[test])

test_results$Scores = score_func(w = w_opt[["w"]],X = review_dt[test])

# Using the scores, calculate the class 1 probabilities and store it in the Probability column
test_results$Probability = sigmoid_func(w = w_opt[["w"]],X = review_dt[test])

# Write a function classify_func that assigns a positive sentiment
# if the class 1 probability is greater than or equal to .5 or a negative sentiment otherwise.
# Apply this function to the Probability column and store the results in a new column called Sentiment_predicted
#Classify function
classify_func = function(X){
  if(X>=.5){
    return(1)
  } else{ 
    return(-1)
  }
}
classify_func = Vectorize(classify_func,"X")

test_results$Sentiment_predicted = classify_func(X = test_results$Probability)

# Assessing Predictions
# Using the ID column, attach the corresponding reviews and ratings to test_result datatable
test_results = cbind(test_results,products[test_results$ID,.(rating,review)])

# Quiz Question 2: How many reviews in the test set were classified as having a negative sentiment?
sum(test_results$Scores<0)

# Let's look at the 10 reviews that our model has the highest confidence are positive reviews
positiveReviews = test_results[order(-Probability)][1:10]
positiveReviews[1]

# Let's look at the 10 reviews that our model has the highest confidence are negative reviews
negativeReviews = test_results[order(Probability)][1:10]
negativeReviews[1]

# Quiz Question 3: Explain the following:
#   1.	Why do some of the reviews that our model has the highest confidence are positive, were actually negative?
#   2.	Why do some of the reviews that our model has the highest confidence are negative, were actually positive?
# Answer of 1: By looking at some of samples of those highly predicted to be positive, it seems the comments are neutral or a story,
#              and not actually an evaluation.
# Answer of 2: By looking at some of samples of those highly predicted to be negative, 

# Calculating Accuracy:
# Write a function (accuracy_func) to calculate the percentage
# of time the predicted sentiment was the same as the true sentiment.
# Calculate the accuracy of the model
accuracy_func = function(y,yhat){
  return(sum(y==yhat)/length(y))
}

#Accuracy on the test set
accuracy_func(y = test_results$Sentiment,yhat = test_results$Sentiment_predicted)

#Accuracy on the training set
accuracy_func(y = products$sentiment[train],yhat = classify_func(sigmoid_func(w = w_opt[["w"]],X = review_dt[train])))

# Quiz Question 4: What is the accuracy of the model on the test data?
# Answer: It is 0.6270602
  
# Quiz Question 5: What is the accuracy of the model on the training data?
# Answer: It is 0.6303886
  
# Quiz Question 6: Does a higher accuracy on the training data always imply that one classifier is better than another?
# Answer: No it doesn't. Higher accuracy on the training data than on test data might imply overfitting

# Train a new classifier using the review title:

# Train a new classified using only the review title. Please complete the following steps:
#   1. Compute the document term matrix
#   2. Calculate the optimal weights
#   3. Use the weights to generate predictions

#New doc-term matrix for review titles
review_title_dt = create_doc_term_dt(text = products$review_summary,max_sparsity = .999)  

#New weights
w_opt_title=train_func(w = runif(n = ncol(review_title_dt[train]),min = -1,max = 1),
                       X = review_title_dt,lr = .1,
                       y = products$sentiment,iters = 100)
#New Scores
test_results$Scores_Title = score_func(w = w_opt_title[["w"]],X = review_title_dt[test])

#Probability Predictions
test_results$Probability_Title = sigmoid_func(w = w_opt_title[["w"]],X = review_title_dt[test])
test_results$Sentiment_predicted_Title = classify_func(X = test_results$Probability_Title)
test_results = cbind(test_results,products[test_results$ID,.(Title=review_summary)])

#Training Accuracy
accuracy_func(y = products$sentiment[train],yhat = classify_func(sigmoid_func(w = w_opt_title[["w"]],X = review_title_dt[train])))

#Test Accuracy
accuracy_func(y = products$sentiment[test],yhat = classify_func(sigmoid_func(w = w_opt_title[["w"]],X = review_title_dt[test])))

# Quiz Question 7: For a new review, which of the two models would you pick for sentiment classification?
# Answer: the first one

# Baseline Performance: Majority Class Prediction:

# Lastly, we will compare the performance of our models against the baseline performance of a majority class classifier.
# The majority classifier predicts the most common class for all reviews.
# Ideally the model you develop should be more accurate than the majority class classifier.
# Calculate the accuracy of the majority class classifier
# Class 1 is the most common class
accuracy_func(y = products$sentiment[test],yhat = 1)

# Quiz Question 8: How reliable are any of the models we just developed?
# Answer: Not very reliable. The model is biased since alsmost 80% of the sentiments are positive in test (i.e. yhat = 1)

# Quiz Question 9: Can you think of simple ways by which we could improve our models?
# Answer: Add more data and run it for more iterations.

# Finally, the above model is based on unigram and we should also try bigram or more words to see if the robustness and accuracy
# might be better. Also, Logistic Regression is just one of many algorithms out there.