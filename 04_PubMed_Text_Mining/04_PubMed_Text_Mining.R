# ===========================================================================
# MACHINE LEARNING & BIG DATA IN PRECISION MEDICINE
# Script: Automated Text Mining and NLP on Scientific Literature (PubMed)
# Method: Corpus Tokenization, Term-Document Matrix, and Network Analysis
# Author: Dr. Foca Marco Carchedi
# Date: 2026-03-04
# ===========================================================================

library(SnowballC)
library(tm)           # Core Text Mining framework
library(wordcloud)    # Wordcloud generation
library(RColorBrewer) # Color palettes
library(igraph)       # Network analysis and co-occurrence graphs
library(tidyverse)    # Data manipulation and ggplot2
library(SnowballC)    # To perform stemming

# ---------------------------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------------------------
# Load dataset containing PubMed search results (Topic: Magnetic Hydrogels) 
data_pubmed <- read.csv("csv-Magnetichy-set.csv", 
                        stringsAsFactors = FALSE,
                        fileEncoding = "UTF-8")

# ---------------------------------------------------------------------------
# 2. CORPUS CREATION
# ---------------------------------------------------------------------------
# Transform the 'Title' column into a volatile Corpus for NLP processing
my_corpus <- VCorpus(VectorSource(data_pubmed$Title))

# ---------------------------------------------------------------------------
# 3. NLP PRE-PROCESSING PIPELINE (Cleaning & Normalization)
# ---------------------------------------------------------------------------
# Convert all text to lowercase
my_corpus <- tm_map(my_corpus, content_transformer(tolower))

# Remove punctuation and numbers (noise reduction)
my_corpus <- tm_map(my_corpus, removePunctuation)
my_corpus <- tm_map(my_corpus, removeNumbers)

# Define and remove stopwords (standard English + domain/search specific terms)
custom_stop <- c(stopwords("english"), 
                 "magnetic", "hydrogel", "hydrogels", 
                 "based", "using", "study", "new", "prepared")
my_corpus <- tm_map(my_corpus, removeWords, custom_stop)

# Stemming: reduce words to their root form (e.g., "properties" -> "properti")
my_corpus <- tm_map(my_corpus, stemDocument)

# Final cleanup: strip whitespace generated during transformations
my_corpus <- tm_map(my_corpus, stripWhitespace)

# ---------------------------------------------------------------------------
# 4. DOCUMENT-TERM MATRIX (DTM) GENERATION
# ---------------------------------------------------------------------------
# Create the DTM, filtering out words shorter than 4 characters
dtm <- DocumentTermMatrix(my_corpus, 
                          control = list(wordLengths = c(4, Inf)))

# ---------------------------------------------------------------------------
# 5. SPARSITY REDUCTION
# ---------------------------------------------------------------------------
# Remove rare terms to focus the analysis on the most statistically significant concepts
dtm <- removeSparseTerms(dtm, 0.99)

# ---------------------------------------------------------------------------
# 6. FREQUENCY CALCULATION
# ---------------------------------------------------------------------------
# Convert to matrix to calculate total occurrences for each term across the corpus
m <- as.matrix(dtm)
v <- sort(colSums(m), decreasing = TRUE)
d <- tibble(word = names(v), freq = v)

# ---------------------------------------------------------------------------
# 7A. FREQUENCY ANALYSIS (BARPLOT)
# ---------------------------------------------------------------------------
# Visualize top terms exceeding a specific frequency threshold
freq_plot <- d %>% 
  filter(freq > 70) %>% 
  ggplot(aes(x = reorder(word, freq), y = freq)) + 
  geom_bar(stat = 'identity', aes(fill = freq)) + 
  scale_fill_gradient(low = "skyblue", high = "#0F243E") +
  coord_flip() + 
  labs(
    title = "Top Frequent Terms in Scientific Literature", 
    subtitle = "Corpus: PubMed Abstracts on Magnetic Hydrogels",
    x = "Stemmed Terms", 
    y = "Total Occurrences"
  ) +
  theme_minimal() + 
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "white", color = NA),  
    panel.background = element_rect(fill = "white", color = NA), 
    panel.grid.minor = element_blank(),                          
    axis.line = element_line(colour = "black")                   
  )

print(freq_plot)
ggsave('01_word_frequencies_barplot.png', plot = freq_plot, width = 8, height = 6, dpi = 300) 

# ---------------------------------------------------------------------------
# 7B. TERM ASSOCIATION ANALYSIS
# ---------------------------------------------------------------------------
# Identify strong statistical correlations (min 0.2) for the top 5 most frequent terms
top_terms <- d$word[1:5] 
assoc_list <- lapply(top_terms, function(t) findAssocs(dtm, t, 0.2))
names(assoc_list) <- top_terms
print("--- Term Associations (Correlation > 0.2) ---")
print(assoc_list)

# ---------------------------------------------------------------------------
# 7C. CO-OCCURRENCE NETWORK ANALYSIS
# ---------------------------------------------------------------------------
# Create an adjacency matrix based on term co-presence in documents
dtm_reduced <- dtm[, findFreqTerms(dtm, lowfreq = 60)]
termMatrix <- t(as.matrix(dtm_reduced)) %*% (as.matrix(dtm_reduced))

# Build the undirected graph (removing self-loops and simplifying)
g <- graph_from_adjacency_matrix(termMatrix, weighted = TRUE, mode = "undirected") %>% 
  igraph::simplify()

# Aesthetic parameters for the network graph
V(g)$label  <- V(g)$name
V(g)$degree <- degree(g)

jpeg("02_co_occurrence_network.jpeg", width = 8, height = 6, units = 'in', res = 300)
par(mar = c(0, 0, 0, 0))
plot(g, 
     layout = layout.kamada.kawai,
     vertex.size = V(g)$degree * 0.7, 
     vertex.label.cex = 0.8,
     vertex.label.color = "black",
     vertex.color = "#FFD54F",
     edge.width = E(g)$weight / max(E(g)$weight) * 4)
dev.off()

# ---------------------------------------------------------------------------
# 7D. [cite_start]WORDCLOUD GENERATION [cite: 134]
# ---------------------------------------------------------------------------
set.seed(123) 
png("03_wordcloud_pubmed.png", width = 1200, height = 800, res = 250)
wordcloud(words = d$word,
          freq = d$freq, 
          min.freq = 5,
          max.words = 100,
          random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"), 
          scale = c(2.5, 0.5)) 
dev.off()

cat("\nNLP pipeline completed successfully. Plots saved to the Working Directory.\n")

