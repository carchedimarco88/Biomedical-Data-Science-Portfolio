library(tidyverse)
library(foreign)
library(caret)
library(mice)
library(miceadds)
library(here)
library(progressr)
library(digest)

# Impostazione barra di avanzamento
handlers(list(
  handler_progress(
    format   = ":spin :current/:total [:bar] :percent in :elapsed ETA: :eta",
    width    = 60,
    complete = "+"
  )
))

# Caricamento dati
tbi_raw <- read.spss(here('data', 'TBI.sav'),
                     use.value.labels = FALSE,
                     to.data.frame = TRUE)

# Pulizia variabili ridondanti
var_to_retain <- setdiff(names(tbi_raw), 
                         c("pupil.i", "glucoset", "sodiumt", "hbt"))
tbi_raw <- tbi_raw[var_to_retain]

# Funzioni di controllo tipo dati
is_logical_var <- function(col) {
  non_missing <- col[!is.na(col)]
  all(non_missing %in% 0:1)
}

is_integer_var <- function(col) {
  non_missing <- col[!is.na(col)]
  !is.logical(col) && all(non_missing == as.integer(non_missing))
}

# Trasformazione finale
tbi <- tbi_raw %>% 
  mutate(
    across(where(is_logical_var), as.logical),
    across(where(is_integer_var), as.integer)
  )

# Output per il test
print(digest::digest(tbi))

# Generazione Fold
set.seed(1234)
n_folds <- 5
folds <- createFolds(tbi$d.unfav, k = n_folds, returnTrain = TRUE)

# Verifica quinto fold
print(folds[[5]][1])

seed_mice <- 1234

m <- 10
max_iter <- 20
visit_sequence <- "monotone" # imputation ordered accordingly from low
# to high proportion of missing data
# Usiamo quelli di default che sono, come si legge 
# dall'help `?mice`:
#   - pmm, predictive mean matching, for numeric data
#   - logreg, logistic regression imputation, for binary data
#     (factor with 2 levels)
#   - polyreg, polytomous regression imputation, for unordered 
#     categorical data (factor > 2 levels) 
#   - polr, proportional odds model, for ordered categrorical data 
#     (factor > 2 levels).
custom_mice <- function(db) {
  mice(db,
       m = m, maxit = max_iter,
       visitSequence = visit_sequence,
       seed = seed_mice,
       printFlag = FALSE
  )
}  
tbi_mice <- custom_mice(tbi)

# check convergence
plot(tbi_mice)
# check the distributions
densityplot(tbi_mice)

# Calcolo delle prestazioni del modello applicando i coefficienti mediati (pooled) 

coef_pooled_model <- function(train_mice_db) {
  mod <- with(train_mice_db, glm(
    d.unfav ~ trial + age + hypoxia + hypotens + tsah + d.pupil + d.motor + ctclass,
    family = binomial
  ))
  
  pooled_coef <- summary(pool(mod))$estimate
  pooled_name <- summary(pool(mod))$term
  
  setNames(pooled_coef, pooled_name)
}

get_unfav_acc <- function(coefs, test_db) {
  
  aux_db <- test_db
  
  aux_db$trial75 <- test_db$trial == 75
  aux_db$hypoxiaTRUE <- test_db$hypoxia
  aux_db$hypotensTRUE <- test_db$hypotens
  aux_db$tsahTRUE <- test_db$tsah
  
  var_used <- setdiff(names(coefs), "(Intercept)")
  
  aux_db <- as.matrix(aux_db[var_used])
  
  estimated_unfav <- as.vector(
    coefs["(Intercept)"] + aux_db %*% coefs[-1]
  )
  
  predictions <- plogis(estimated_unfav) > 0.5
  mean(test_db$d.unfav == predictions)
}

get_pooled_acc <- function(db_mice) {
  coefs <- coef_pooled_model(db_mice$train)
  
  m <- db_mice$train$m
  
  accs <- map_dbl(seq_len(m), ~{
    test_db <- complete(db_mice$test, .x)
    get_unfav_acc(coefs, test_db)
  })
  
  mean(accs)
}

get_k_sets <- function(mice_db, k, folds_list) {
  # seq_len crea una sequenza lunga quanto il dataset originale
  in_train <- seq_len(nrow(mice_db$data)) %in% folds_list[[k]]
  in_test <- !in_train
  
  list(
    # subset_datlist è una funzione di miceadds per filtrare oggetti mids
    train = subset_datlist(mice_db, in_train, toclass="mids"),
    test = subset_datlist(mice_db, in_test, toclass="mids")
  )
}

# Funzione per calcolare l'accuratezza del primo approccio (MI-CV)
compute_micv <- function(db_mice, folds_list) {
  p <- progressr::progressor(length(folds_list))
  map_dbl(seq_along(folds_list), ~{
    # capture.output serve a non intasare la console
    capture.output(db_k <- get_k_sets(db_mice, .x, folds_list))
    res <- get_pooled_acc(db_k)
    p() # aggiorna barra
    res
  })
}

# Esecuzione calcolo accuratezza approccio 1
with_progress({
  micv_accs <- compute_micv(tbi_mice, folds)
})

# Visualizzazione risultati approccio 1
print(micv_accs)
boxplot(micv_accs, main = "Accuratezza MI-CV")

# --- Implementazione Approccio 2 (CV-MI) ---

create_imputed_k_folds <- function(db, folds_list) {
  p <- progressr::progressor(steps = 2 * length(folds_list))
  
  map(folds_list, ~{
    # Imputazione separata per il training set del fold corrente
    k_train <- custom_mice(db[.x, , drop = FALSE])
    p()
    
    # Imputazione separata per il test set del fold corrente
    k_test <- custom_mice(db[-.x, , drop = FALSE])
    p()
    
    list(train = k_train, test = k_test)
  })
}

# Esecuzione (Attenzione: questo comando effettua 2000 cicli totali)
with_progress({
  k_mice_tbi <- create_imputed_k_folds(tbi, folds)
})

# Controllo della struttura
str(k_mice_tbi, 2)


# 1. Calcolo accuratezza CV-MI (Imputazione interna)
cvmi_accs <- map_dbl(k_mice_tbi, get_pooled_acc)

# 2. Visualizzazione distribuzione accuratezza
print(cvmi_accs)
boxplot(cvmi_accs, main = "Distribuzione Accuratezza CV-MI")

# 3. Funzione per il boxplot comparativo
paired_boxplot <- function(micv_accs, cvmi_accs) {
  data.frame(
    micv = micv_accs,
    cvmi = cvmi_accs
  ) %>% 
    pivot_longer(everything(),
                 names_to = "Method",
                 values_to = "Accuracy"
    ) %>% 
    ggplot(aes(y = Accuracy, x = Method, fill = Method)) +
    geom_boxplot() +
    theme_bw() +
    labs(title = "Confronto MI-CV vs CV-MI")
}

# 4. Generazione del grafico finale
paired_boxplot(micv_accs, cvmi_accs)
