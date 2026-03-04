# ===========================================================================
# MACHINE LEARNING & BIG DATA IN PRECISION MEDICINE 
# Script: Handling Missing Data in Clinical Trials (TBI Study)
# Method: Multiple Imputation (MICE) & Cross-Validation Strategies (MI-CV vs CV-MI)
# Author: Dr. Foca Marco Carchedi
# Date: 2026-03-04
# ===========================================================================

library(tidyverse)
library(foreign)
library(caret)
library(mice)
library(miceadds)
library(here)
library(progressr)
library(digest)

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP & DATA INGESTION
# ---------------------------------------------------------------------------
# Configure progress bar for long imputation iterations
handlers(list(
  handler_progress(
    format   = ":spin :current/:total [:bar] :percent in :elapsed ETA: :eta",
    width    = 60,
    complete = "+"
  )
))

# Load Traumatic Brain Injury (TBI) dataset
tbi_raw <- read.spss(here('data', 'TBI.sav'),
                     use.value.labels = FALSE,
                     to.data.frame = TRUE)

# Remove redundant or uninformative variables to optimize imputation
var_to_retain <- setdiff(names(tbi_raw), 
                         c("pupil.i", "glucoset", "sodiumt", "hbt"))
tbi_raw <- tbi_raw[var_to_retain]

# Helper functions to strictly define data types for imputation
is_logical_var <- function(col) {
  non_missing <- col[!is.na(col)]
  all(non_missing %in% 0:1)
}

is_integer_var <- function(col) {
  non_missing <- col[!is.na(col)]
  !is.logical(col) && all(non_missing == as.integer(non_missing))
}

# Final data transformation pipeline
tbi <- tbi_raw %>% 
  mutate(
    across(where(is_logical_var), as.logical),
    across(where(is_integer_var), as.integer)
  )

# Verify data integrity via hash
print(digest::digest(tbi))

# ---------------------------------------------------------------------------
# 2. CROSS-VALIDATION SETUP
# ---------------------------------------------------------------------------
# Setting up folds for 5-fold cross validation. Target variable: d.unfav 
# (Unfavorable outcome: 1 = Yes, 0 = No)
set.seed(1234)
n_folds <- 5
folds <- createFolds(tbi$d.unfav, k = n_folds, returnTrain = TRUE)

# ---------------------------------------------------------------------------
# 3. BASELINE IMPUTATION: MULTIPLE IMPUTATION (MICE)
# ---------------------------------------------------------------------------
seed_mice <- 1234
m <- 10             # Number of multiple imputations
max_iter <- 20      # Maximum iterations per imputation
visit_sequence <- "monotone" # Ordered from low to high missingness

# Default MICE algorithms used:
# - pmm (predictive mean matching) for numeric data
# - logreg (logistic regression) for binary data
# - polyreg (polytomous regression) for unordered categorical > 2 levels

custom_mice <- function(db) {
  mice(db,
       m = m, maxit = max_iter,
       visitSequence = visit_sequence,
       seed = seed_mice,
       printFlag = FALSE
  )
}  
tbi_mice <- custom_mice(tbi)

# Convergence and distribution checks
jpeg("01_mice_convergence_plot.jpeg", width = 8, height = 6, units = "in", res = 300)
plot(tbi_mice)
dev.off()

jpeg("02_mice_density_plot.jpeg", width = 8, height = 6, units = "in", res = 300)
densityplot(tbi_mice)
dev.off()

# ---------------------------------------------------------------------------
# 4. MODEL EVALUATION FUNCTIONS (POOLING ESTIMATES)
# ---------------------------------------------------------------------------
# Function to calculate pooled coefficients from logistic regression
coef_pooled_model <- function(train_mice_db) {
  mod <- with(train_mice_db, glm(
    d.unfav ~ trial + age + hypoxia + hypotens + tsah + d.pupil + d.motor + ctclass,
    family = binomial
  ))
  
  pooled_coef <- summary(pool(mod))$estimate
  pooled_name <- summary(pool(mod))$term
  
  setNames(pooled_coef, pooled_name)
}

# Function to calculate out-of-sample accuracy on imputed test sets
get_unfav_acc <- function(coefs, test_db) {
  aux_db <- test_db
  
  # Map variables according to logistic regression output
  aux_db$trial75 <- test_db$trial == 75
  aux_db$hypoxiaTRUE <- test_db$hypoxia
  aux_db$hypotensTRUE <- test_db$hypotens
  aux_db$tsahTRUE <- test_db$tsah
  
  var_used <- setdiff(names(coefs), "(Intercept)")
  aux_db <- as.matrix(aux_db[var_used])
  
  # Calculate linear predictor and convert to probabilities
  estimated_unfav <- as.vector(
    coefs["(Intercept)"] + aux_db %*% coefs[-1]
  )
  
  predictions <- plogis(estimated_unfav) > 0.5
  mean(test_db$d.unfav == predictions)
}

# Main function to get pooled accuracy
get_pooled_acc <- function(db_mice) {
  coefs <- coef_pooled_model(db_mice$train)
  m <- db_mice$train$m
  
  accs <- map_dbl(seq_len(m), ~{
    test_db <- complete(db_mice$test, .x)
    get_unfav_acc(coefs, test_db)
  })
  
  mean(accs)
}

# ---------------------------------------------------------------------------
# 5. APPROACH 1: MI-CV (Multiple Imputation THEN Cross-Validation)
# ---------------------------------------------------------------------------
# WARNING: This approach can induce data leakage as imputation is performed 
# on the full dataset before splitting into Train and Test sets.

get_k_sets <- function(mice_db, k, folds_list) {
  in_train <- seq_len(nrow(mice_db$data)) %in% folds_list[[k]]
  in_test <- !in_train
  
  list(
    train = subset_datlist(mice_db, in_train, toclass="mids"),
    test = subset_datlist(mice_db, in_test, toclass="mids")
  )
}

compute_micv <- function(db_mice, folds_list) {
  p <- progressr::progressor(length(folds_list))
  map_dbl(seq_along(folds_list), ~{
    capture.output(db_k <- get_k_sets(db_mice, .x, folds_list))
    res <- get_pooled_acc(db_k)
    p() 
    res
  })
}

# Execute Approach 1
with_progress({
  micv_accs <- compute_micv(tbi_mice, folds)
})
print("MI-CV Accuracy:")
print(micv_accs)

# ---------------------------------------------------------------------------
# 6. APPROACH 2: CV-MI (Cross-Validation THEN Multiple Imputation)
# ---------------------------------------------------------------------------
# CORRECT METHOD: Imputation is strictly isolated within the training fold 
# to prevent any data leakage into the test set.

create_imputed_k_folds <- function(db, folds_list) {
  p <- progressr::progressor(steps = 2 * length(folds_list))
  
  map(folds_list, ~{
    # Isolate training fold and impute
    k_train <- custom_mice(db[.x, , drop = FALSE])
    p()
    
    # Isolate test fold and impute (separately!)
    k_test <- custom_mice(db[-.x, , drop = FALSE])
    p()
    
    list(train = k_train, test = k_test)
  })
}

# Execute Approach 2 (Computationally intensive: 2000 total iterations)
cat("Running isolated CV-MI protocol. This may take several minutes...\n")
with_progress({
  k_mice_tbi <- create_imputed_k_folds(tbi, folds)
})

cvmi_accs <- map_dbl(k_mice_tbi, get_pooled_acc)
print("CV-MI Accuracy:")
print(cvmi_accs)

# ---------------------------------------------------------------------------
# 7. COMPARATIVE VISUALIZATION (MI-CV vs CV-MI)
# ---------------------------------------------------------------------------
paired_boxplot <- function(micv_accs, cvmi_accs) {
  data.frame(
    `MI-CV (Leakage Risk)` = micv_accs,
    `CV-MI (Strict Protocol)` = cvmi_accs
  ) %>% 
    pivot_longer(everything(),
                 names_to = "Method",
                 values_to = "Accuracy"
    ) %>% 
    ggplot(aes(x = Method, y = Accuracy, fill = Method)) +
    geom_boxplot(alpha = 0.8) +
    scale_fill_manual(values = c("MI-CV (Leakage Risk)" = "#FF6565", 
                                 "CV-MI (Strict Protocol)" = "#0F243E")) +
    labs(
      title = "Clinical Missing Data Handling: MI-CV vs. CV-MI",
      subtitle = "Comparing optimistic accuracy (Leakage) vs. rigorous protocol (Strict)",
      y = "Out-of-Sample Accuracy",
      x = "Imputation Methodology",
      caption = "Analysis and visualization: Dr. Marco Carchedi"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(face = "bold")
    )
}

final_plot <- paired_boxplot(micv_accs, cvmi_accs)
print(final_plot)
ggsave("03_imputation_strategy_comparison.jpeg", plot = final_plot, width = 8, height = 6, dpi = 300)

cat("\nScript completed successfully. All plots have been saved to the Working Directory.\n")
