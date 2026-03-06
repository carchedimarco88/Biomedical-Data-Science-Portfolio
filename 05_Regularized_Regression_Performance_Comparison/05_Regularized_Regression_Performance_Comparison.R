# ===========================================================================
# MACHINE LEARNING & BIG DATA IN PRECISION MEDICINE 
# Script: Predictive Performance Comparison of Regression Models
# Method: Out-of-sample evaluation via RMSE
# Author: Dr. Foca Marco Carchedi
# Date: 2026-03-04
# ===========================================================================
library(tidyverse)
library(glmnet)

# Data Loading and Cleaning
wd <- read_csv("wd.txt", skip = 1) %>% 
  dplyr::select(-c(Date, `Activity Calories`)) %>% 
  rename(Calories = `Calories Burned`)

# Train/Test Split (70% - 30%)
# ------------------------------------------------------------------------------
# METHODOLOGICAL NOTE: We perform the split BEFORE creating the matrices
# to prevent any form of data leakage.
set.seed(123) 
n <- nrow(wd)
train_idx <- sample(1:n, n * 0.7)

wd_train <- wd[train_idx, ]
wd_test  <- wd[-train_idx, ]

# Model Training
# ------------------------------------------------------------------------------

# --- A. OLS (Ordinary Least Squares) ---
model_ols <- lm(Calories ~ ., data = wd_train)
pred_ols  <- predict(model_ols, newdata = wd_test)
rmse_ols  <- sqrt(mean((wd_test$Calories - pred_ols)^2))

# --- Matrix Preparation for GLMNET ---
# We create the matrices separately for the training and test sets
# The model.matrix function transforms the predictors into numerical/matrix format
x_train <- model.matrix(Calories ~ ., wd_train)[, -1]
y_train <- wd_train$Calories

x_test <- model.matrix(Calories ~ ., wd_test)[, -1]
y_test <- wd_test$Calories

# --- B. LASSO (Alpha = 1) ---
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
best_lam_lasso <- cv_lasso$lambda.min
pred_lasso <- predict(cv_lasso, s = best_lam_lasso, newx = x_test)
rmse_lasso <- sqrt(mean((y_test - pred_lasso)^2))

# --- C. RIDGE (Alpha = 0) ---
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)
best_lam_ridge <- cv_ridge$lambda.min
pred_ridge <- predict(cv_ridge, s = best_lam_ridge, newx = x_test)
rmse_ridge <- sqrt(mean((y_test - pred_ridge)^2))

# --- D. ELASTIC NET (Alpha = 0.5 ) ---
# We set alpha = 0.5 for an equal balance between L1 and L2 penalties.
cv_enet <- cv.glmnet(x_train, y_train, alpha = 0.5)
best_lam_enet <- cv_enet$lambda.min
pred_enet <- predict(cv_enet, s = best_lam_enet, newx = x_test)
rmse_enet <- sqrt(mean((y_test - pred_enet)^2))

# RESULTS VERIFICATION AND COEFFICIENTS
# ------------------------------------------------------------------------------

# We show the critical coefficients to see the variable selection
print("--- COEFFICIENT COMPARISON: Steps and Distance ---")

print("1. OLS (Note negative Steps and huge Distance):")
print(coef(model_ols)[c("Steps", "Distance")])

print("2. LASSO:")
coef_lasso <- coef(cv_lasso, s = best_lam_lasso)
# We extract the corresponding coefficients (handling names with backticks if needed)

print(coef_lasso[rownames(coef_lasso) %in% c("Steps", "Distance"), , drop=FALSE])

# Performance Comparison and Output
# ------------------------------------------------------------------------------
df_res <- data.frame(
  Model = c("OLS", "Ridge", "LASSO", "Elastic Net"),
  RMSE = c(rmse_ols, rmse_ridge, rmse_lasso, rmse_enet)
)

print("--- RMSE RANKING (Lower is better) ---")
print(df_res[order(df_res$RMSE), ])

# Plot
p <- ggplot(df_res, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(RMSE, 2)), family = "sans", fontface = "bold", vjust = -0.5, size = 4) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Predictive Performance Comparison",
       subtitle = "Out-of-sample evaluation via RMSE",
       y = "RMSE (Calories - kcal)",
       x = "Regression Models",
       caption = "Analysis and visualization by Dr. Carchedi Foca Marco") +
  theme_minimal() +
  scale_fill_manual(values = c(
    "LASSO"       = "#0F243E", 
    "Elastic Net" = "#808080", 
    "Ridge"       = "#FFD54F", 
    "OLS"         = "#FF6565"  
  ))  +
  theme(legend.position = "none",
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "aliceblue", color = NA),
        axis.title.y = element_text(face = "bold", family = "sans", size = 11, color = "black"),
        axis.title.x = element_text(face = "bold", family = "sans", size = 11, color = "black", margin = margin(t = 10)),
        axis.line.y = element_line(color = "gray30", size = 0.5),
        axis.text.y = element_text(color = "black", family = "sans"),
        axis.line.x = element_line(color = "gray30", size = 0.5),
        axis.text.x = element_text(color = "black", face = "bold", family = "sans", size = 13),
        plot.title = element_text(family = "sans", face = "bold", size = 14, hjust = 0),
        plot.subtitle = element_text(family = "sans", size = 11, color = "gray10", hjust = 0),
        plot.caption = element_text(hjust = 1, face = "italic", color = "gray20", size = 10),
        panel.grid.major.y = element_line(color = "gray92", size = 0.5),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank()
  )

print(p)
ggsave("performance_comparison_Dr_Carchedi_Foca_Marco.jpeg", plot = p, width = 8, height = 6)

# The messages "Option grouped=FALSE enforced... since < 3 observations per fold" 
# appear only because the dataset is very small (30 rows total). 
# By splitting into training and test sets, few data points remain for cross-validation.
# It is a technical warning irrelevant to the conceptual validity of the educational exercise.
