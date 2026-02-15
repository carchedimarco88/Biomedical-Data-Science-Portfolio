library(tidyverse)
library(caret)
library(kknn)

set.seed(1)
n_cases <- 50
n_predictors <- 5000

# nomi delle variabili
predictors_names <- set_names(
  paste0("var_", seq_len(n_predictors))
)

# generazione dei predittori
predictors <- map_df(
  predictors_names,
  function(x) rnorm(n = n_cases)   # indipendente da x
)

predictors

# risposta binaria
response <- rbinom(n = n_cases, size = 1, prob = 0.5)

response

# dataset finale
df <- bind_cols(
  response = as_factor(response),
  predictors
)

df

set.seed(1)           # seed per la casualità della generazione dei fold
n_folds <- 10
folds <- createFolds(df$response, k = n_folds, returnTrain = TRUE)
folds

setdiff(1:50, folds$Fold05)

shrink_to_top_corr <- function(df, n_pred_to_retain = 100) {
  # extract components
  resp <- as.numeric(df[["response"]])
  pred <- select(df, -response)
  
  # get correlation and corresponding names
  correlations <- cor(resp, pred) 
  pred_names <- colnames(correlations)
  
  # convert correlation matrix in an ordered named vector
  correlations <- set_names(as.vector(correlations), pred_names) %>% 
    sort(decreasing = TRUE)
  
  # get top n
  top_correlated <- correlations[seq_len(n_pred_to_retain)] %>%
    names()
  df[c("response", top_correlated)]
}

df_wrong <- shrink_to_top_corr(df)
df_wrong


get_biased_accuracy <- function(data, train_index) {
  train_set <- data[train_index, ]
  test_set  <- data[-train_index, ]
  
  model <- kknn(response ~ ., 
                      train = train_set, 
                      test = test_set, k = 7)
  
  mean(model$fitted.values == test_set$response)  # Accuratezza
}

biased_accuracies <- map_dbl(folds, get_biased_accuracy, data = df_wrong)
biased_accuracies

get_fair_accuracy <- function(data, train_index) {
  train_set <- data[train_index, ]                     
  test_set  <- data[-train_index, ]
  
  top_train <- shrink_to_top_corr(train_set)           
  
  top_test <- test_set[names(top_train)]               
  
  model <- kknn(response ~ ., 
                train = top_train,                     
                test  = top_test,                      
                k = 7)                                 
  
  mean(model$fitted.values == test_set$response)       
}

fair_accuracies <- map_dbl(folds, get_fair_accuracy, data = df)   
fair_accuracies

data_frame(
  biased = biased_accuracies,
  fair = fair_accuracies
) %>% 
  boxplot()
title(paste(
  "Folds: ", n_folds, ";",
  "Top ", 100, "predictors."
))
