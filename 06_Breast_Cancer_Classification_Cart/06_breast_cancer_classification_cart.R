# ===========================================================================
# MACHINE LEARNING & BIG DATA IN PRECISION MEDICINE
# Script: Predictive Analysis of Breast Tumor Behavior
# Model: Decision Tree Classification (CART) with 10-fold Cross-Validation
# Author: Dr. Foca Marco Carchedi
# Date: 2026-02-22
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP AND LIBRARY LOADING
# ---------------------------------------------------------------------------
library(caret)      
library(rpart)      
library(tidyverse)  
library(tree)       
library(here)       
library(digest)     

# ---------------------------------------------------------------------------
# 2. DATA INGESTION AND TIDYING (DATA PREPARATION)
# ---------------------------------------------------------------------------
data_path <- here("data", "diagnosis.data")

# Definition of variable metadata (Clinical glossary)
var_names <- c(
  "id", "clump_thickness", "uniform_cell_size", "uniform_cell_shape",
  "marginal_adhesion", "single_epith_cell_size", "bare_nuclei", 
  "bland_chromatin", "normal_nucleoli", "mitoses", "class"
)

# Initial data import
diagnosis <- read_csv(data_path, col_names = var_names, na = "?")

# Verify correct import using hash
digest::digest(diagnosis)

# Optimization of data ingestion
var_type <- "ciiiiiiiiii"  

# Loading with explicit handling of missing values ("?")
diagnosis <- data_path %>% 
  read_csv(
    col_types = var_type,
    col_names = var_names,
    na = "?"  
  ) %>% 
  mutate(
    # Convert target to factor: Class 2 -> benign, Class 4 -> malignant
    class = factor(class, levels = c(2, 4), labels = c("benign", "malignant"))
  )

diagnosis

# Data integrity check via hash (Expected: 2508cbfd8b36b1b001eea856bb84e665)
cat("--- DATA INTEGRITY CHECK ---\n")
cat("Digest Hash:", digest::digest(diagnosis), "\n")
print(table(diagnosis$class))

# ---------------------------------------------------------------------------
# 3. STRATIFIED PARTITIONING (TRAIN/TEST SPLIT)
# ---------------------------------------------------------------------------
set.seed(1) # Ensure reproducibility of the split

# Create indices for the training set (300 observations)
train_idx <- createDataPartition(
  diagnosis$class, 
  p = 300 / nrow(diagnosis)
)[[1L]]  

length(train_idx)

# Verify class proportions post-split
class_proportion <- list(
  original = table(diagnosis$class) / nrow(diagnosis),
  train = table(diagnosis$class[train_idx]) / length(train_idx)
)

class_proportion

# Define the training set
train_set <- diagnosis[train_idx, , drop = FALSE]
train_set

# Define the testing set
test_set <- diagnosis[-train_idx, , drop = FALSE]
test_set

# Create the training dataset for tree modeling (excluding ID)
train_set_tree <- train_set %>% select(-id)

# ---------------------------------------------------------------------------
# 4. EXPLORATORY MODEL (PACKAGE 'TREE')
# ---------------------------------------------------------------------------
tree_model <- tree(class ~ ., data = train_set_tree)

# Print model summary
cat("\n--- BASELINE MODEL SUMMARY (tree) ---\n")
mod_summary <- summary(tree_model)
print(mod_summary)

# Calculate the number of terminal nodes
nodi_terminali <- mod_summary$size

# Extract misclassification rate
misclass_rate <- mod_summary$misclass[1] / mod_summary$misclass[2]

# Calculate final sum rounded to 3 decimal places
somma_finale <- round(nodi_terminali + misclass_rate, 3)
cat("Number of terminal nodes:", nodi_terminali, "\n")
cat("Misclassification error rate:", round(misclass_rate, 3), "\n")
cat("-----\n")
cat("The sum is:", somma_finale, "\n")

# --- Visualization and Export of the Exploratory Tree ---
jpeg("Exploratory_Tree_Baseline_Carchedi.jpeg", width = 8, height = 6, units = 'in', res = 300)
par(mar = rep(4, 4))
plot(tree_model)
text(tree_model, cex = 0.8)
title("Classification Tree for breast tumor behavior - Dr. Carchedi", line = 2)
dev.off()

# Predictions on the test set
previsioni_test <- predict(tree_model, newdata = test_set, type = "class")

# Generate confusion matrix
conf_matrix <- confusionMatrix(data = previsioni_test, reference = test_set$class)

# Confusion matrix report
print(conf_matrix)

# Extract Balanced Accuracy
balanced_accuracy <- conf_matrix$byClass["Balanced Accuracy"]
cat("-----\n")
cat("Exact Balanced Accuracy value:", round(balanced_accuracy, 4), "\n")

# ---------------------------------------------------------------------------
# 5. OPTIMIZED MODEL: 10-FOLD CV AND CP TUNING (PACKAGE 'RPART')
# ---------------------------------------------------------------------------
# Search for optimal Bias-Variance trade-off via Cross-Validation
set.seed(1)
tune_grid <- expand.grid(
  # Linear sequence between 0.04 and 0.06
  .cp = seq(from = 4e-2, to = 6e-2, length.out = 50) 
)

# Training with caret
caret_rpart <- train(
  class ~ .,
  data = select(train_set, -id),
  method    = "rpart",
  metric    = "Accuracy",
  trControl = trainControl(
    method = 'cv',
    number = 10
  ),
  na.action = na.rpart,  # Specific missing values handling for rpart
  tuneGrid  = tune_grid
)

# --- Visualization and Export of Bias-Variance Plot ---
jpeg("Tuning_CV_Accuracy_Carchedi.jpeg", width = 8, height = 6, units = 'in', res = 300)
plot(caret_rpart, main = "Complexity Parameter (CP) Tuning - Dr. Carchedi")
dev.off()

# Visualization and Export of the final model (Best Tree)
jpeg("Best_10-fold_CV_Behavior_Carchedi.jpeg", width = 8, height = 6, units = 'in', res = 300)
plot(caret_rpart$finalModel, uniform = TRUE, compress = TRUE, margin = 0.06, branch = 0.5)
text(caret_rpart$finalModel, use.n = TRUE, cex = 0.8)
title("Best 10-fold CV Classification Tree - Dr. Carchedi")
dev.off()

cat("\nPlot saved successfully in the working directory.\n")

caret_rpart$finalModel

# ---------------------------------------------------------------------------
# 6. FINAL PERFORMANCE EVALUATION AND EXPORT
# ---------------------------------------------------------------------------

# Predictions on the test set
previsioni_test_rpart <- predict(caret_rpart, newdata = test_set, na.action = na.pass)
conf_matrix_rpart <- confusionMatrix(data = previsioni_test_rpart, reference = test_set$class)
print(conf_matrix_rpart)

# Positive Predictive Value for the "benign" class
ppv_benign <- conf_matrix_rpart$byClass["Pos Pred Value"]
cat("-----\n")
cat("The Positive Predictive Value (benign) is:", round(ppv_benign, 4), "\n")
cat("\nScript completed successfully.\nNote: Plots have been saved to the Working Directory.")
