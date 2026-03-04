# ===========================================================================
# MACHINE LEARNING & BIG DATA IN PRECISION MEDICINE
# Script: High-Dimensional Feature Selection and Data Leakage Prevention
# Method: K-Nearest Neighbors (KNN), Biased vs. Fair Cross-Validation
# Author: Dr. Foca Marco Carchedi
# Date: 2026-03-04
# ===========================================================================

library(tidyverse)
library(caret)
library(kknn)

# ---------------------------------------------------------------------------
# 1. DATA SIMULATION (High-Dimensional Biomedical Scenario)
# ---------------------------------------------------------------------------
# Simulating a scenario typical of genomics/transcriptomics:
# Few clinical cases (n=50) but thousands of predictors/genes (n=5000)
set.seed(1)
n_cases <- 50
n_predictors <- 5000

# Generate predictor names
predictors_names <- set_names(
  paste0("var_", seq_len(n_predictors))
)

# Generate pure noise for predictors (independent from the target)
predictors <- map_df(
  predictors_names,
  function(x) rnorm(n = n_cases)   
)

# Generate a random binary clinical response (e.g., Treatment Success/Failure)
response <- rbinom(n = n_cases, size = 1, prob = 0.5)

# Build the final simulated dataset
df <- bind_cols(
  response = as_factor(response),
  predictors
)

# ---------------------------------------------------------------------------
# 2. CROSS-VALIDATION SETUP
# ---------------------------------------------------------------------------
set.seed(1)            # Ensure fold generation reproducibility
n_folds <- 10
folds <- createFolds(df$response, k = n_folds, returnTrain = TRUE)

# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION FUNCTION
# ---------------------------------------------------------------------------
# Function to retain only the top 'n' predictors most correlated with the response
shrink_to_top_corr <- function(df, n_pred_to_retain = 100) {
  # Extract components
  resp <- as.numeric(df[["response"]])
  pred <- select(df, -response)
  
  # Calculate correlations and extract names
  correlations <- cor(resp, pred) 
  pred_names <- colnames(correlations)
  
  # Convert correlation matrix into an ordered named vector
  correlations <- set_names(as.vector(correlations), pred_names) %>% 
    sort(decreasing = TRUE)
  
  # Extract the names of the top N correlated predictors
  top_correlated <- correlations[seq_len(n_pred_to_retain)] %>%
    names()
  
  # Return the reduced dataframe
  df[c("response", top_correlated)]
}

# ---------------------------------------------------------------------------
# 4. BIASED APPROACH (DATA LEAKAGE)
# ---------------------------------------------------------------------------
# ERROR: Applying feature selection to the ENTIRE dataset before Cross-Validation.
# This causes Data Leakage, as the test sets will contain information already 
# used to select the features, leading to falsely optimistic accuracy.

df_leaked <- shrink_to_top_corr(df)

get_biased_accuracy <- function(data, train_index) {
  train_set <- data[train_index, ]
  test_set  <- data[-train_index, ]
  
  model <- kknn(response ~ ., 
                train = train_set, 
                test = test_set, 
                k = 7)
  
  mean(model$fitted.values == test_set$response)  # Return Accuracy
}

# Calculate biased accuracies across all folds
biased_accuracies <- map_dbl(folds, get_biased_accuracy, data = df_leaked)

# ---------------------------------------------------------------------------
# 5. FAIR APPROACH (STRICT METHODOLOGY)
# ---------------------------------------------------------------------------
# CORRECT: Applying feature selection strictly WITHIN each training fold.
# The test set remains completely unseen during the selection process.

get_fair_accuracy <- function(data, train_index) {
  train_set <- data[train_index, ]                     
  test_set  <- data[-train_index, ]
  
  # Feature selection applied ONLY on the training fold
  top_train <- shrink_to_top_corr(train_set)            
  
  # Align test set features with the ones selected in the training fold
  top_test <- test_set[names(top_train)]                
  
  model <- kknn(response ~ ., 
                train = top_train,                     
                test  = top_test,                      
                k = 7)                                 
  
  mean(model$fitted.values == test_set$response)        
}

# Calculate fair accuracies across all folds
fair_accuracies <- map_dbl(folds, get_fair_accuracy, data = df)   

# ---------------------------------------------------------------------------
# 6. RESULTS VISUALIZATION
# ---------------------------------------------------------------------------
# Compare the falsely optimistic biased accuracy with the realistic fair accuracy

results_df <- tibble(
  Biased = biased_accuracies,
  Fair = fair_accuracies
) %>% 
  pivot_longer(cols = everything(), names_to = "Methodology", values_to = "Accuracy")

# Plotting the comparison
comparison_plot <- ggplot(results_df, aes(x = Methodology, y = Accuracy, fill = Methodology)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 16, outlier.size = 2) +
  scale_fill_manual(values = c("Biased" = "#FF6565", "Fair" = "#0F243E")) +
  labs(
    title = "Data Leakage Impact on Predictive Accuracy",
    subtitle = paste("10-Fold CV | Top 100 predictors retained out of 5000 random variables"),
    x = "Cross-Validation Methodology",
    y = "Model Accuracy",
    caption = "Analysis and visualization: Dr. Marco Carchedi"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold")
  )

print(comparison_plot)
ggsave("02_data_leakage_bias_vs_fair_boxplot.jpeg", plot = comparison_plot, width = 8, height = 6, dpi = 300)

cat("\nScript completed successfully. Notice how the 'Biased' approach finds false signal in pure noise.\n")
