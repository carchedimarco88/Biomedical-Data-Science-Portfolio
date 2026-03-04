library(tidyverse)
library(splines)

set.seed(2701)

generated <- function( n ){
  x <- sort(runif(n, 0, 8))
  eps <- rnorm(n, 0, 0.5)
  x <- sort(runif(n, 0, 8))
  eps <- rnorm(n, 0, 0.5)
  y <- sin(x) + eps
  # return:
  tibble(x = x, y = y)
}
n  <- 1000
xy <- generated( n )
xy[[2]][[123]]

# 3. Funzione di fitting 
fit_spline_with_df <- function(db, df) {
  lm(y ~ ns(x, df), data = db)
}
# 4.a Ciclo for 
df_of_interest <- 1:40
models_for <- vector("list", length(df_of_interest)) 
for (i in seq_along(df_of_interest)) {
  models_for[[i]] <- fit_spline_with_df(xy, i) 
} 
names(models_for) <- df_of_interest

#4.b Funzionale -  Approccio Funzionale (con purrr::map)
df_of_interest <- 1:40
models_functional <- map(df_of_interest,
                         ~ fit_spline_with_df(db = xy, df = .x) # Completato con la funzione di fit
)
names(models_functional) <- df_of_interest

#5 Verifica che i due metodi siano equivalenti
all.equal(models_for, models_functional)

# 6 Estrazione del risultato richiesto
risultato <- models_functional[[36]]$coefficients["(Intercept)"]
print(risultato)

# Costruzione del dataframe con tutte le predizioni delle spline
splines_predictions <- imap_dfr(models_functional,
                                
                                ~ tibble(
                                  x  = xy[["x"]],
                                  y_real = xy[["y"]],
                                  y_estimated  = .x[["fitted.values"]],
                                  error = y_estimated - y_real,
                                  df = factor(.y, levels = df_of_interest)
                                )
)

# Aggiunta della curva generatrice (sin(x))
xy[["generating_y"]] <- sin(xy[["x"]])

# Plot base con dati reali e curva sinusoidale

base_plot <- xy %>% 
  ggplot(aes(x = x)) +
  geom_point(aes(y = y), colour = "blue", shape = 16, alpha = 0.1) +
  geom_line(aes(y = generating_y), linetype = "dashed")

# Plot delle spline per ogni grado di libertà
base_plot +
  geom_line(
    data = splines_predictions,
    aes(x = x, y = y_estimated),
    colour = "red"
  ) +
  facet_wrap(~ df, ncol = 4)


#________________EXTRA: Procedura quantitativa per definire il modello migliore

# Fold per la Cross-Validation 
set.seed(123)
xy_cv <- xy %>% mutate(fold = sample(rep(1:10, length.out = n())))

# Griglia dei gradi di libertà
df_values <- 1:40

# Funzione per calcolare l'errore di un singolo modello su un fold specifico
cv_results <- map_dfr(df_values, function(d) {
  
  # Per ogni grado di libertà, cicliamo sui 10 fold 
  errors <- map_dbl(1:10, function(f) {
    train_data <- xy_cv %>% filter(fold != f)
    test_data  <- xy_cv %>% filter(fold == f)
    
    # Fit del modello sui dati di training
    model <- lm(y ~ ns(x, df = d), data = train_data)
    
    # Predizione su nuovi dati (test) 
    preds <- predict(model, newdata = test_data)
    
    # Calcolo del Root Mean Squared Error
    sqrt(mean((test_data$y - preds)^2))
  })
  
  # Restituiamo il valore medio dell'errore per quel DF
  tibble(df = d, mean_rmse = mean(errors))
})

# Identifichiamo il miglior DF
best_df <- cv_results$df[which.min(cv_results$mean_rmse)]
print(paste("Il miglior numero di gradi di libertà secondo la 10-fold CV è:", best_df))

#Visualizzazione
cv_results %>% 
  ggplot(aes(df, mean_rmse)) +
  geom_line() +
  geom_point() +
  theme_minimal()

