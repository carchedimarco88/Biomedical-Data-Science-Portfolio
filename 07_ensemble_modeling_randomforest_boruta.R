# ===========================================================================
# MACHINE LEARNING & BIG DATA 
# Script: Predictive Modeling with Random Forest and Boruta Feature Selection
# Method: Conditional Inference Trees (CTREE), Bagging, and Random Forest
# Author: Dr. Foca Marco Carchedi
# Date: 2026-03-01
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP AND DATA INGESTION
# ---------------------------------------------------------------------------
library(caret)
library(partykit)
library(randomForest)
library(Boruta)
library(tidyverse)
library(here)

# Load dataset
bicycles <- readRDS(here("data", "bicycles.rds"))

# Check for missing values across all features
map_int(bicycles, ~ sum(is.na(.x)))

# Stratified Train/Test split (75/25)
p <- 300 / 400
set.seed(1234)
train_idx <- createDataPartition(bicycles$sales, p = p, list = FALSE)

train_set <- bicycles[train_idx, ]
test_set <- bicycles[-train_idx, ]

# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
# PLOT 1: Target Variable (Sales) Distribution Check
plot_density <- tibble(
  sales = c(train_set$sales, test_set$sales),
  set = c(
    rep("train", nrow(train_set)),
    rep("test", nrow(test_set))
  )
) %>% 
  ggplot(aes(sales, fill = set)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distributions of Target Variable (Sales) within Train and Test Sets")

print(plot_density) 
ggsave("01_target_density.jpeg", plot = plot_density, width = 8, height = 6, dpi = 300) 

# ---------------------------------------------------------------------------
# 3. BASELINE MODEL: CONDITIONAL INFERENCE DECISION TREE (CIDT)
# ---------------------------------------------------------------------------
set.seed(1234)
tune_grid <- expand.grid(
  mincriterion = seq(from = 0.95, to = 0.99, length.out = 100)
)
n_cv <- 5

# Ensure reproducibility within cross-validation folds
cidt_seed <- c(
  map(seq_len(n_cv),                         
      ~ nrow(tune_grid) * (.x - 1) +  
        seq_len(nrow(tune_grid))      
  ),
  nrow(tune_grid) * n_cv + 1      
)

# Train the CIDT model
cidt_model <- train(
  x = train_set %>% select(-sales),
  y = train_set$sales,
  method = "ctree",
  metric = "RMSE",
  trControl = trainControl(
    method = "cv",
    number = n_cv,
    seeds = cidt_seed
  ),
  tuneGrid = tune_grid
)

# PLOT 2: CIDT Cross-Validation Tuning Profile
jpeg("02_cidt_tuning.jpeg", width = 800, height = 600, res = 100)
plot(cidt_model, main = "Cross-Validation RMSE Profile (CIDT)")
dev.off()

# Extract the final optimized tree
cidt_model$finalModel

# PLOT 3: Final CIDT Tree Visualization
jpeg("03_cidt_final_tree.jpeg", width = 1200, height = 800, res = 100) 
plot(cidt_model$finalModel, main = "Final Conditional Inference Tree")
dev.off()

# ---------------------------------------------------------------------------
# 4. METRIC EVALUATION FUNCTION
# ---------------------------------------------------------------------------
# Helper function to extract and calculate train/test RMSE for each model
model_res <- function(model) {
  train_rmse <- min(model$results$RMSE)
  
  predictions <- predict(model, test_set %>% select(-sales))
  test_rmse <- sqrt(mean((predictions - test_set[["sales"]])^2))
  
  tibble(
    method = model$method,
    train = train_rmse,
    test = test_rmse,
    tune_name = names(model$bestTune),
    tune_value = model$bestTune[[1]]
  )
}

cidt_res <- model_res(cidt_model) 
print("--- CIDT Model Results ---")
print(cidt_res)

# ---------------------------------------------------------------------------
# 5. ENSEMBLE MODELING: BAGGING
# ---------------------------------------------------------------------------
set.seed(1234)
bagg_seed <- c(
  map(seq_len(n_cv), ~ (.x - 1) + 1),  
  n_cv + 1
)

# Train Bagging model (Random Forest with mtry = total number of predictors)
bagg_model <- train(
  x = train_set %>% select(-sales),
  y = train_set$sales,
  method = "rf",
  metric = "RMSE",
  trControl = trainControl(
    method = "cv",
    number = n_cv,
    seeds = bagg_seed
  ),
  tuneGrid = data.frame(mtry = length(train_set) - 1)
)

bagg_res <- model_res(bagg_model) %>% 
  mutate(method = "Bagging")

print("--- Bagging Model Results ---")
print(bagg_res)

# PLOT 4: Variable Importance (Bagging)
jpeg("04_bagging_varimp.jpeg", width = 800, height = 600, res = 100)
varImpPlot(bagg_model$finalModel, main = "Variable Importance (Bagging)")
dev.off()

# ---------------------------------------------------------------------------
# 6. ENSEMBLE MODELING: TRADITIONAL RANDOM FOREST
# ---------------------------------------------------------------------------
set.seed(1234)
tune_grid <- expand.grid(
  mtry = seq_len(ncol(train_set) - 1)
)

rf_seed <- c(
  map(seq_len(n_cv),                        
      ~ nrow(tune_grid) * (.x - 1) +  
        seq_len(nrow(tune_grid))      
  ),
  nrow(tune_grid) * n_cv + 1      
)

# Train the Random Forest model with mtry tuning
rf_model <- train(
  x = train_set %>% select(-sales),
  y = train_set[["sales"]],
  method = "rf",
  metric = "RMSE",
  trControl = trainControl(
    method = "cv",
    number = n_cv,
    seeds = rf_seed
  ),
  tuneGrid = tune_grid
)

# PLOT 5: Random Forest Cross-Validation Tuning Profile
jpeg("05_rf_tuning.jpeg", width = 800, height = 600, res = 100)
plot(rf_model, main = "Cross-Validation RMSE Profile (Random Forest)")
dev.off()

# Evaluate and store Random Forest results 
rf_res <- model_res(rf_model) 
print("--- Random Forest Results ---")
print(rf_res)

# PLOT 6: Variable Importance (Random Forest)
jpeg("06_rf_varimp.jpeg", width = 800, height = 600, res = 100)
varImpPlot(rf_model$finalModel, main = "Variable Importance (Random Forest)")
dev.off()

# ---------------------------------------------------------------------------
# 7. ADVANCED FEATURE SELECTION: BORUTA ALGORITHM
# ---------------------------------------------------------------------------
# Implement Boruta to isolate statistically significant predictors
boruta_model <- Boruta(sales ~ ., data = train_set, maxRuns = 500)

# PLOT 7: Boruta Feature Importance Boxplots
jpeg("07_boruta_importance.jpeg", width = 1000, height = 600, res = 100)
plot(boruta_model, main = "Boruta Feature Importance Analysis")
dev.off()

print("--- Boruta Selection Summary ---")
print(boruta_model)

# ---------------------------------------------------------------------------
# 8. OPTIMIZED RANDOM FOREST (BORUTA-SELECTED FEATURES)
# ---------------------------------------------------------------------------
# Extract only "Confirmed" significant features
boruta_res <- boruta_model$finalDecision
boruta_best <- names(boruta_res[boruta_res == "Confirmed"])

set.seed(1234)
tune_grid <- expand.grid(
  mtry = seq_along(boruta_best)
)

rf_seed <- c(
  map(seq_len(n_cv),                        
      ~ nrow(tune_grid) * (.x - 1) +  
        seq_len(nrow(tune_grid))      
  ),
  nrow(tune_grid) * n_cv + 1      
)

# Train the final RF model using only selected variables
boruta_rf_model <- train(
  x = train_set %>% select(all_of(boruta_best)),
  y = train_set[["sales"]],
  method = "rf",
  metric = "RMSE",
  trControl = trainControl(
    method = "cv",
    number = n_cv,
    seeds = rf_seed
  ),
  tuneGrid = tune_grid
)

# PLOT 8: Tuning Profile for Boruta-optimized RF
jpeg("08_boruta_rf_tuning.jpeg", width = 800, height = 600, res = 100)
plot(boruta_rf_model, main = "Cross-Validation RMSE Profile (Boruta + RF)")
dev.off()

boruta_rf_res <- model_res(boruta_rf_model) %>% 
  mutate(method = "Boruta + rf")
print("--- Boruta + Random Forest Results ---")
print(boruta_rf_res)

# PLOT 9: Variable Importance (Boruta + RF)
jpeg("09_boruta_rf_varimp.jpeg", width = 800, height = 600, res = 100)
varImpPlot(boruta_rf_model$finalModel, main = "Variable Importance (Boruta + RF)")
dev.off()

# ---------------------------------------------------------------------------
# 9. PERFORMANCE COMPARISON AND FINAL VISUALIZATION
# ---------------------------------------------------------------------------
# Aggregate all model results
res <- bind_rows(
  cidt_res,
  bagg_res,
  rf_res,
  boruta_rf_res
)

# PLOT 10: Final ggplot comparing Train vs. Test RMSE across models
plot_final <- res %>%
  mutate(method = recode(method,
                         "ctree" = "CTREE",
                         "Bagging" = "BAGGING",
                         "rf" = "RF",
                         "Boruta + rf" = "BORUTA + RF")) %>%
  mutate(method = as.factor(method)) %>%
  pivot_longer(
    cols = c("train", "test"), 
    names_to = "set",
    values_to = "RMSE"
  ) %>%
  mutate(set = toupper(set)) %>% 
  
  ggplot(aes(x = method, y = RMSE, color = set)) +
  geom_point(size = 3) + 
  
  scale_color_manual(values = c("TRAIN" = "#FFC000", 
                                "TEST"  = "#0F243E")) +
  
  labs(title = "Algorithm Comparison: Train vs Test Set RMSE",
       subtitle = "Evaluation of generalizability and overfitting across model architectures",
       x = "PREDICTIVE ALGORITHM", 
       y = "ROOT MEAN SQUARE ERROR (RMSE)",
       color = "DATASET",
       caption = "Analysis and visualization: Dr. Marco Carchedi") + 
  
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "aliceblue", color = NA),
    axis.title.y = element_text(face = "bold", family = "sans", size = 11, color = "black"),
    axis.title.x = element_text(face = "bold", family = "sans", size = 11, color = "black", 
                                margin = ggplot2::margin(t = 10)), 
    axis.line.y = element_line(color = "gray30", linewidth = 0.5), 
    axis.text.y = element_text(color = "black", family = "sans"),
    axis.line.x = element_line(color = "gray30", linewidth = 0.5), 
    axis.text.x = element_text(color = "black", family = "sans", size = 12),
    plot.title = element_text(family = "sans", face = "bold", size = 14, hjust = 0),
    plot.subtitle = element_text(family = "sans", size = 11, color = "gray10", hjust = 0),
    plot.caption = element_text(hjust = 1, face = "italic", color = "gray20", size = 10),
    plot.caption.position = "plot",
    panel.grid.major.y = element_line(color = "gray80", linewidth = 0.5), 
    panel.grid.major.x = element_line(color = "gray80", linewidth = 0.5),
    panel.grid.minor = element_line(color = "gray80", linewidth = 0.5)
  )

print(plot_final)
ggsave("10_Train_vs_Test_RMSE_Comparison.jpeg", plot = plot_final, width = 8, height = 6, dpi = 300)

cat("\nScript completed successfully.\nNote: All performance evaluation plots have been saved to the Working Directory.\n")