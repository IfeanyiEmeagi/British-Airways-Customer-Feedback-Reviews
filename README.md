# Forage Work Simulation Project - British Airways Customer Feedback Reviews

The Forage work simulation project provides real world tasks to both experienced and fresh graduates to tried their hands on. In this task, I scrapped British Airways customer reviews hosted on Skytrax[https://www.airlinequality.com/airline-reviews/british-airways/page/1/?sortby=post_date%3ADesc&pagesize=10] and applied topic modelling using Latent Dirichlet Allocation (LDA) to identify the most prevalent issues the customers are saying.

### Tasks Implemented

- Create a web scrapper using Scrapy Python Library. The scrapper crawls the web page and scraps the title, review and score assigned by both verified and unverified customers. The scrapping process starts from the first page and loops through the pagination till the end of the page.
- The scrapped data is saved locally as a JSON formatted file.
