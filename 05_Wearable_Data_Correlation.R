# ------------------------------------------------------------------------------
# HOMEWORK WEEK-5 MODULO 2
# CONFRONTO DELLE PERFORMANCE PREDITTIVE DEI MODELLI DI REGRESSIONE
# Autore: Dott. Carchedi Foca Marco
# Data: 15/02/2026
# ------------------------------------------------------------------------------

library(tidyverse)
library(glmnet)

# Caricamento e Pulizia Dati
wd <- read_csv("wd.txt", skip = 1) %>% 
  dplyr::select(-c(Date, `Activity Calories`)) %>% 
  rename(Calories = `Calories Burned`)

# Split Train/Test (70% - 30%)
# ------------------------------------------------------------------------------
# NOTA METODOLOGICA: Eseguiamo lo split PRIMA di creare le matrici
# per evitare qualsiasi forma di data leakage.
set.seed(123) 
n <- nrow(wd)
train_idx <- sample(1:n, n * 0.7)

wd_train <- wd[train_idx, ]
wd_test  <- wd[-train_idx, ]

# Addestramento Modelli
# ------------------------------------------------------------------------------

# --- A. OLS (Ordinary Least Squares) ---
model_ols <- lm(Calories ~ ., data = wd_train)
pred_ols  <- predict(model_ols, newdata = wd_test)
rmse_ols  <- sqrt(mean((wd_test$Calories - pred_ols)^2))

# --- Preparazione Matrici per GLMNET ---
# Creiamo le matrici separatamente per training e test set
# La funzione model.matrix trasforma i predittori in formato numerico/matrice
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
# Impostiamo alpha = 0.5 per un bilanciamento equo tra penalità L1 e L2.
cv_enet <- cv.glmnet(x_train, y_train, alpha = 0.5)
best_lam_enet <- cv_enet$lambda.min
pred_enet <- predict(cv_enet, s = best_lam_enet, newx = x_test)
rmse_enet <- sqrt(mean((y_test - pred_enet)^2))

# VERIFICA DEI RISULTATI E COEFFICIENTI
# ------------------------------------------------------------------------------

# Mostriamo i coefficienti critici per vedere la selezione delle variabili
print("--- CONFRONTO COEFFICIENTI: Steps e Distance ---")

print("1. OLS (Notare Steps negativo e Distance enorme):")
print(coef(model_ols)[c("Steps", "Distance")])

print("2. LASSO:")
coef_lasso <- coef(cv_lasso, s = best_lam_lasso)
# Estraiamo i coefficienti corrispondenti (gestendo i nomi con backtick se serve)

print(coef_lasso[rownames(coef_lasso) %in% c("Steps", "Distance"), , drop=FALSE])

# Confronto Performance e Output
# ------------------------------------------------------------------------------
df_res <- data.frame(
  Modello = c("OLS", "Ridge", "LASSO", "Elastic Net"),
  RMSE = c(rmse_ols, rmse_ridge, rmse_lasso, rmse_enet)
)

print("--- CLASSIFICA RMSE (Minore è meglio) ---")
print(df_res[order(df_res$RMSE), ])

# Grafico
p <- ggplot(df_res, aes(x = reorder(Modello, RMSE), y = RMSE, fill = Modello)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(RMSE, 2)), family = "sans", fontface = "bold", vjust = -0.5, size = 4) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Confronto delle Performance di Generalizzazione",
       subtitle = "Valutazione out-of-sample tramite RMSE",
       y = "RMSE (Calorie - kcal)",
       x = "Modelli di regressione",
       caption = "Analisi e visualizzazione a cura del dott. Carchedi Foca Marco") +
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
ggsave("confronto_performance_dott._Carchedi_Foca_Marco.jpeg", plot = p, width = 8, height = 6)

# I messaggi "Option grouped=FALSE enforced... since < 3 observations per fold" 
# compaiono solo perché il dataset è piccolissimo (30 righe totali). 
# Dividendo in training e test, rimangono pochi dati per la cross-validation.
# È un avviso tecnico irrilevante per la validità concettuale dell'esercizio didattico.