# **British Airways Data Science Job Simulation on Forage**

This job simulation consists of two main tasks:

- **Customer Review Insights**
- **Predictive Modeling on Customer Buying Behavior**

## **Customer Review Insights**

This task involves scraping British Airways customer reviews from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways/page/1/?sortby=post_date%3ADesc&pagesize=10), cleaning and preprocessing the data, and applying **unsupervised machine learning techniques**, specifically **Latent Dirichlet Allocation (LDA)**, to identify the most prevalent customer concerns. The findings are presented in a **single PowerPoint slide**.

**_Slide Image:_** _(Include slide image here)_

### **Tasks Implemented**

- Developed a **web scraper** using the **Scrapy** Python library to extract customer reviews, including the review title, text, and score.
- The scraper loops through all pagination pages until the last review page is reached.
- Cleaned and preprocessed the data by **removing non-alphabetical characters** and converting text to **lowercase** for normalization.
- Tokenized reviews and **removed stopwords** and words with fewer than two characters.
- **Lemmatized** words to reduce them to their root form.
- Created a **Gensim dictionary** and **corpus** from the processed reviews.
- Trained an **LDA model** using the dictionary and corpus.
- Initially experimented with **five topics**, but due to overlapping insights, **three topics** were selected as the optimal number.
- **Visualized** insights using the **pyLDAviz** library.

**_Insights Visualization:_** _(Include visualization image here)_

---

## **Predictive Model on Customer Buying Behavior**

This task involves building a predictive model to determine whether a customer will **complete the booking process** or not. The dataset contains **50,000 customer observations** with **14 features**, including the target variable.

- The **target variable** consists of two classes: **0 (did not complete booking)** and **1 (completed booking)**.
- The dataset is **highly imbalanced**, with most customers **not completing** the booking process (Class 0).
- The dataset includes **four categorical columns**, one of which, `'route'`, has **high cardinality** with **799 unique categories**.

### **Tasks Implemented**

- Identified the **encoding format** of the dataset to ensure proper reading.
- Mapped **weekdays** to numerical values, starting with **Monday as Day 1**.
- Converted **object data types** to **categorical** format.
- Split the dataset into **training and test sets**.
- **Oversampled** the training set using **Synthetic Minority Over-sampling Technique (SMOTE)** to handle class imbalance.
- Chose **XGBoost** for its **support for categorical features** and **strong performance on tabular data**.
- Used **Grid Search** to identify the **best model parameters**.
- Evaluated the model on the **test dataset**.
- **Graphically visualized** feature importance and the **confusion matrix**.

### **Results Presentation:**

_(Insert slide/image of results here)_
