# Biomedical Data Science & Machine Learning Portfolio
R code for Clinical Data Analysis, Machine Learning, and Text Mining applied to Pharmaceutical and Healthcare contexts

## About Me
Hi, I'm **Foca Marco Carchedi**.
I am a **Healthcare Analytics Engineer** and **Data Analyst** based in Italy, specializing in the intersection of healthcare and data science. My background combines clinical expertise (Pharmacy degree, clinical monitoring) with advanced technical skills in R, Machine Learning, and Big Data analysis.

Currently, I am expanding my expertise through a Master's degree in *Machine Learning and Big Data in Precision Medicine and Biomedical Research* at the University of Padova.

## Project Overview
This repository contains a collection of R projects developed to solve practical problems in the biomedical and pharmaceutical domains. The focus is on applying statistical learning methods to real-world clinical scenarios, from data imputation in Traumatic Brain Injury (TBI) studies to text mining of PubMed literature.

## Tech Stack & Skills
* **Languages:** R, SQL
* **Core Libraries:** `tidyverse`, `caret`, `mice`, `ggplot2`, `tm` (Text Mining), `splines`.
* **Key Concepts:**
    * Missing Data Imputation (MICE algorithm)
    * Supervised Learning (Regression, Classification)
    * Feature Selection & Bias/Variance Trade-off
    * Biomedical Text Mining & NLP
    * Wearable Data Analysis

---

## Projects & Case Studies

### 1. Clinical Data Imputation for TBI Studies (`03_TBI_Clinical_Data_Imputation.R`)
**Objective:** Handling missing data in clinical datasets is critical for regulatory compliance and study validity.
* **Context:** Analyzed a dataset of Traumatic Brain Injury (TBI) patients.
* **Method:** Implemented the **MICE (Multivariate Imputation by Chained Equations)** algorithm to handle missing values in predictors like hypoxia, hypotension, and CT scans.
* **Outcome:** Compared "MI-CV" vs "CV-MI" strategies to ensure unbiased model performance estimation.
* *Keywords: Clinical Trials, Data Cleaning, MICE, Logistic Regression.*

### 2. PubMed Text Mining on Hydrogels (`04_PubMed_Text_Mining.R`)
**Objective:** Extracting trends and insights from unstructured scientific literature.
* **Context:** Analyzed a corpus of scientific abstracts related to "Magnetic Hydrogels" extracted from PubMed.
* **Method:** Used Natural Language Processing (NLP) techniques: tokenization, stop-word removal, stemming, and n-gram analysis.
* **Outcome:** Generated correlation networks and word clouds to identify key research clusters and emerging technologies in pharmaceutical formulation.
* *Keywords: NLP, Text Mining, Pharmaceutical R&D, Unstructured Data.*

### 3. Feature Selection & Model Bias (`02_Feature_Selection_and_Bias.R`)
**Objective:** Demonstrating the risks of data leakage and overfitting in high-dimensional biological data.
* **Method:** Simulated a dataset with 5000 predictors to test feature selection strategies. Compared "Biased" accuracy (selecting features on the whole dataset) vs "Fair" accuracy (selecting within Cross-Validation folds).
* *Keywords: Overfitting, Cross-Validation, High-Dimensional Data.*

### 4. Non-Linear Modeling with Splines (`01_Spline_Regression_Analysis.R`)
**Objective:** Modeling complex biological relationships that represent non-linear patterns.
* **Method:** Fitted B-Splines with varying degrees of freedom to capture non-linear signal in noisy data using functional programming (`purrr`).

### 5. Wearable Device Data Analysis (`05_Wearable_Data_Correlation.R`)
**Objective:** Analyzing correlation patterns in physiological data from wearable sensors.
* **Method:** Spearman correlation analysis on real-world activity tracker data to identify relationships between calories burned and movement intensity.

---

### Contact
I am open to opportunities in **Pharmaceutical Data Science**, **Clinical Data Analysis**, and **Hospital Data Management**.

* **Location:** Cosenza/Rende, Italy
* **Role:** Data Analyst / Pharmacist
