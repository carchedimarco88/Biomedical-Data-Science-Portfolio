# --- PubMed_Text_Mining - Foca Marco Carchedi ---

# Caricamento librerie necessarie per l'analisi del testo e visualizzazione
library(tm)           # Text Mining
library(wordcloud)    # Nuvola di parole
library(RColorBrewer) # Palette colori
library(igraph)       # Network analysis
library(tidyverse)    # Data manipulation e ggplot2

# 1. IMPORTAZIONE DATI
# Caricamento del dataset esportato da PubMed (ricerca su Magnetic Hydrogels)
data_pubmed <- read.csv("csv-Magnetichy-set.csv", 
                        stringsAsFactors = FALSE,
                        fileEncoding = "UTF-8")

# 2. CREAZIONE DEL CORPUS
# Trasformazione della colonna 'Title' in un Corpus volatile per l'elaborazione
my_corpus <- VCorpus(VectorSource(data_pubmed$Title))

# 3. PRE-PROCESSING (Pulizia e Normalizzazione)
# Trasformazione in minuscolo
my_corpus <- tm_map(my_corpus, content_transformer(tolower))
# Rimozione rumore: punteggiatura e numeri
my_corpus <- tm_map(my_corpus, removePunctuation)
my_corpus <- tm_map(my_corpus, removeNumbers)

# Definizione e rimozione stopwords (comuni + termini specifici del set di ricerca)
custom_stop <- c(stopwords("english"), 
                 "magnetic", "hydrogel", "hydrogels", 
                 "based", "using", "study", "new", "prepared")
my_corpus <- tm_map(my_corpus, removeWords, custom_stop)

# Stemming: riduzione delle parole alla radice (es. properties -> properti)
my_corpus <- tm_map(my_corpus, stemDocument)
# Pulizia finale degli spazi bianchi generati dalle trasformazioni
my_corpus <- tm_map(my_corpus, stripWhitespace)

# 4. DOCUMENT-TERM MATRIX (DTM)
# Creazione della matrice termini-documenti con filtro sulla lunghezza minima parole
dtm <- DocumentTermMatrix(my_corpus, 
                          control = list(wordLengths = c(4, Inf)))

# 5. RIDUZIONE SPARSITÀ
# Rimozione dei termini rari per concentrare l'analisi sui concetti più significativi
dtm <- removeSparseTerms(dtm, 0.99)

# 6. CALCOLO FREQUENZE
# Conversione in matrice per calcolare la somma delle occorrenze per ogni termine
m <- as.matrix(dtm)
v <- sort(colSums(m), decreasing = TRUE)
d <- tibble(word = names(v), freq = v)

# -------------------------------
# 7A. ANALISI FREQUENZE (BARPLOT)
# -------------------------------
# Visualizzazione dei termini con frequenza superiore a una soglia definita
d %>% 
  filter(freq > 70) %>% 
  ggplot(aes(x = reorder(word, freq), y = freq)) + 
  geom_bar(stat = 'identity', aes(fill = freq)) + 
  scale_fill_gradient(low = "skyblue", high = "dodgerblue4") +
  coord_flip() + # Ruoto per leggibilità etichette
  labs(title = "Termini più frequenti", x = "Parole", y = "Occorrenze") +
  theme_minimal()

ggsave('word_frequencies.png') # Salvataggio in formato standard richiesto

# -------------------------------
# 7B. ANALISI DELLE ASSOCIAZIONI
# -------------------------------
# Individuazione delle correlazioni (min 0.2) per i primi 5 termini più frequenti
top_terms <- d$word[1:5] 
assoc_list <- lapply(top_terms, function(t) findAssocs(dtm, t, 0.2))
names(assoc_list) <- top_terms
print(assoc_list)

# -------------------------------
# 7C. NETWORK DI CO-OCCORRENZA
# -------------------------------
# Creazione matrice di adiacenza basata sulla co-presenza dei termini
dtm_reduced <- dtm[, findFreqTerms(dtm, lowfreq = 60)]
termMatrix <- t(as.matrix(dtm_reduced)) %*% (as.matrix(dtm_reduced))

# Costruzione del grafo (rimozione loop e semplificazione)
g <- graph_from_adjacency_matrix(termMatrix, weighted = TRUE, mode = "undirected") %>% 
  igraph::simplify()

# Parametri estetici del grafo
V(g)$label  <- V(g)$name
V(g)$degree <- degree(g)
par(mar = c(0, 0, 0, 0))

plot(g, 
     layout = layout.kamada.kawai,
     vertex.size = V(g)$degree * 0.7, 
     vertex.label.cex = 0.7,
     vertex.label.color = "black",
     vertex.color = "gold",
     edge.width = E(g)$weight / max(E(g)$weight) * 4)

# -------------------------------
# 7D. WORDCLOUD
# -------------------------------
set.seed(123) 
png("wordcloud_pubmed.png", width = 1200, height = 800, res =250)
wordcloud(words = d$word,
          freq = d$freq, 
          min.freq = 5,
          max.words = 100,
          random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"), 
          scale = c(2.5, 0.5)) 

dev.off()
