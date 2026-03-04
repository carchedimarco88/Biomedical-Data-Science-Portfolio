# ===========================================================================
# MACHINE LEARNING & BIG DATA IN PRECISION MEDICINE
# Script: Non-Linear Biological Signal Modeling using B-Splines and purrr
# Method: Spline Regression, Functional Programming, and Cross-Validation
# Author: Dr. Foca Marco Carchedi
# Date: 2026-03-04
# ===========================================================================

library(tidyverse)
library(splines)

set.seed(2701)

# ---------------------------------------------------------------------------
# 1. DATA SIMULATION (Non-linear pattern generation)
# ---------------------------------------------------------------------------
generated <- function(n) {
  x <- sort(runif(n, 0, 8))
  eps <- rnorm(n, 0, 0.5)
  y <- sin(x) + eps
  
  tibble(x = x, y = y)
}

n  <- 1000
xy <- generated(n)

# ---------------------------------------------------------------------------
# 2. MODEL DEFINITION & FUNCTIONAL PROGRAMMING (purrr)
# ---------------------------------------------------------------------------

# Fitting function for natural splines
fit_spline_with_df <- function(db, df) {
  lm(y ~ ns(x, df), data = db)
}

# --- Approach A: Traditional For-Loop ---
df_of_interest <- 1:40
models_for <- vector("list", length(df_of_interest)) 

for (i in seq_along(df_of_interest)) {
  models_for[[i]] <- fit_spline_with_df(xy, i) 
} 
names(models_for) <- df_of_interest

# --- Approach B: Functional Programming (using purrr::map) ---
# This approach ensures cleaner, more efficient, and scalable code
models_functional <- map(df_of_interest,
                         ~ fit_spline_with_df(db = xy, df = .x)
)
names(models_functional) <- df_of_interest

# Verify that both methods yield equivalent results
all.equal(models_for, models_functional)

# Extracting a specific result (e.g., intercept of the 36-DF model)
result <- models_functional[[36]]$coefficients["(Intercept)"]
print(result)

# ---------------------------------------------------------------------------
# 3. PREDICTION EXTRACTION AND VISUALIZATION
# ---------------------------------------------------------------------------

# Building a dataframe containing all spline predictions across different DFs
splines_predictions <- imap_dfr(models_functional,
                                ~ tibble(
                                  x  = xy[["x"]],
                                  y_real = xy[["y"]],
                                  y_estimated  = .x[["fitted.values"]],
                                  error = y_estimated - y_real,
                                  df = factor(.y, levels = df_of_interest)
                                )
)

# Adding the underlying generating curve (sin(x)) for reference
xy[["generating_y"]] <- sin(xy[["x"]])

# Base plot with noisy data and the true underlying signal
base_plot <- xy %>% 
  ggplot(aes(x = x)) +
  geom_point(aes(y = y), colour = "#0F243E", shape = 16, alpha = 0.1) +
  geom_line(aes(y = generating_y), linetype = "dashed", colour = "black", linewidth = 0.8) +
  labs(
    title = "Non-Linear Modeling with Natural Splines",
    x = "Predictor (x)",
    y = "Response (y)"
  ) +
  theme_minimal()

# Faceted plot displaying spline fits for each degree of freedom
spline_grid_plot <- base_plot +
  geom_line(
    data = splines_predictions,
    aes(x = x, y = y_estimated),
    colour = "#FF6565", 
    linewidth = 0.8
  ) +
  facet_wrap(~ df, ncol = 4)

print(spline_grid_plot)
ggsave("01_spline_degrees_of_freedom_grid.jpeg", plot = spline_grid_plot, width = 12, height = 10, dpi = 300)

# ---------------------------------------------------------------------------
# 4. ADVANCED HYPERPARAMETER TUNING (10-FOLD CROSS-VALIDATION)
# ---------------------------------------------------------------------------
# Quantitative procedure to define the optimal model complexity (preventing overfitting)

set.seed(123)
# Assigning Folds for Cross-Validation 
xy_cv <- xy %>% mutate(fold = sample(rep(1:10, length.out = n())))

# Grid for degrees of freedom
df_values <- 1:40

# Calculate the out-of-sample error for each DF across all folds
cv_results <- map_dfr(df_values, function(d) {
  
  # Iterate over the 10 folds for the current degree of freedom 
  errors <- map_dbl(1:10, function(f) {
    train_data <- xy_cv %>% filter(fold != f)
    test_data  <- xy_cv %>% filter(fold == f)
    
    # Model fitting on training data
    model <- lm(y ~ ns(x, df = d), data = train_data)
    
    # Prediction on test set 
    preds <- predict(model, newdata = test_data)
    
    # Calculate Root Mean Squared Error (RMSE)
    sqrt(mean((test_data$y - preds)^2))
  })
  
  # Return the mean cross-validated RMSE for the current DF
  tibble(df = d, mean_rmse = mean(errors))
})

# Identify the optimal DF (minimizing generalization error)
best_df <- cv_results$df[which.min(cv_results$mean_rmse)]
print(paste("The optimal number of degrees of freedom according to 10-fold CV is:", best_df))

# Visualization of the Bias-Variance trade-off
cv_plot <- cv_results %>% 
  ggplot(aes(x = df, y = mean_rmse)) +
  geom_line(colour = "#0F243E", linewidth = 1) +
  geom_point(colour = "#FF6565", size = 2) +
  geom_vline(xintercept = best_df, linetype = "dashed", colour = "red") +
  annotate("text", x = best_df + 2, y = max(cv_results$mean_rmse), 
           label = paste("Optimal DF:", best_df), color = "red", fontface = "bold") +
  labs(
    title = "Cross-Validation RMSE Profile",
    subtitle = "Selecting the optimal spline complexity to minimize generalization error",
    x = "Degrees of Freedom (DF)",
    y = "Mean RMSE (10-fold CV)"
  ) +
  theme_minimal()

print(cv_plot)
ggsave("02_spline_cross_validation_tuning.jpeg", plot = cv_plot, width = 8, height = 6, dpi = 300)

cat("\nScript completed successfully.\nNote: Plots saved to the Working Directory.\n")
