# Required Libraries
install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("e1071")
install.packages("gmodels")

library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(gmodels)

# Load the Dataset
sms_raw <- read.csv("C:\\Users\\User.user\\source\\repos\\57. AI ML p2\\sms_spam.csv", stringsAsFactors = FALSE)

# Examine the Structure
str(sms_raw)

# Convert `type` column to factor
sms_raw$type <- factor(sms_raw$type)
table(sms_raw$type)

# Build a Corpus using `tm` package
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# Clean the Corpus
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords("en"))
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

# Optional: Apply stemming
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

# Create a Document-Term Matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# Split Data into Training and Test Sets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

# Save the Labels
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

# Check Proportions of Spam and Ham
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# Word Cloud Visualization
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

# Create Word Clouds for Spam and Ham
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# Reduce Sparsity in the DTM
sms_dtm_freq_train <- removeSparseTerms(sms_dtm_train, 0.999)

# Find Frequent Terms
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

# Create New DTMs with Frequent Terms Only
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[, sms_freq_words]

# Convert Counts to Factors
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

# Train the Naïve Bayes Model
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# Evaluate Model Performance
sms_test_pred <- predict(sms_classifier, sms_test)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('Predicted', 'Actual'))

# Improve Model with Laplace Smoothing
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)

# Evaluate Improved Model
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('Predicted', 'Actual'))

