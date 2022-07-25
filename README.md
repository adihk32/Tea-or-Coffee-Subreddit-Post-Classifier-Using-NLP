<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Project 3: Classification of Subreddit Posts

---
### Problem Statement:  

- As in-house data scientists with Coftea, the aim of this study is to develop a classification model that will give insights on the top keywords for tea and coffee. The keywords will be used to optimize the recommender system that will be deploy to the website for search bar usage. It will also be used to analyse the social media when we wanted to analyse for sentiment for our products and to classify the post or reviews for the company to improve. The classification model that we developed will rely on the accuracy metric as it is important to identify both tea and coffee products.
---

### Background:
Most people will start their day with a cup of coffee or tea. Whether is a freshly roasted black coffee or a cup English Breakfast, they are like energy booster to the majority. After water, tea and coffee is the most popular drink that people is drinking worldwide. One of the reasons tea and coffee is popular, is the health benefits they bring [[1]](https://www.pennmedicine.org/updates/blogs/health-and-wellness/2019/december/health-benefits-of-tea#:~:text=Numerous%20studies%20have%20shown%20that,lasting%20impact%20on%20your%20wellness.) [[2]](https://www.psypost.org/2022/05/brain-imaging-study-suggests-that-drinking-coffee-enhances-neurocognitive-function-63213).

In 2021, according to Tea Association of the USA inc, [report](https://www.teausa.com/14655/tea-fact-sheet), around 80% of Americans consumed tea and approximately consumed 85 billion servings or 3.9 billion gallons of tea. Coffee Market also predicted with annual increment of 4% [[3]](https://finance.yahoo.com/news/coffee-market-revenue-reach-157-121900670.html).

Furthermore, in a new [report](https://www.grandviewresearch.com/industry-analysis/ready-to-drink-tea-and-ready-to-drink-coffee-market?utm_source=prnewswire&utm_medium=referral&utm_campaign=FMCG_16-May-22&utm_term=ready_to_drink_tea_and_ready_to_drink_coffee_market&utm_content=rd1) by Grand View Research, Inc, the RTD (Ready-To-Drink) Tea and Coffee global market size will increase to USD 167.88 billion by 2030, a 6.2% annual growth prediction. There is also a possibility that Asia Pacific region will hold the largest share of the market.

Coftea, a beverage company launched in 2018 that has various tea and coffee lines that are unique and special for the customers who like to have different experience. It started off as retail shop in Singapore , and now it has various branches within Southeast Asia. Recently, it has started new website that includes online shop to capture the global audience.

### Datasets:
For the datasets, we are going to scrape from the subreddit r/tea and r/Coffee to gain insights of what people are talking about currently. The scraping process will be using PushShift API. As the maximum post allowed for one request is 100 posts, we are building function to iterate through multiple iterations with delay in between iteration.  

We also filter further the posts that are removed or deleted that can be found using `removed_by_category` column and we also drop the posts with no `selftext`. The removed posts and no selftext posts will be save in separate dataframes each.

The datasets that can be found in the data folder:
- `df_tea&coffee_220522.csv` -> combined dataset of r/tea and r/Coffee subreddit that contains 1200 posts from each subreddit
- `df_tea_removed.csv` -> dataset of r/tea that the posts are removed
- `df_tea_notext.csv` -> dataset of r/tea that has no text in the selftext
- `df_coffee_removed.csv` -> dataset of r/Coffee that the posts are removed
- `df_coffee_notext.csv` -> dataset of r/Coffee that has no text in the selftext

---


### Executive Summary: 
The dataset `df_tea&coffee_220522.csv` that is scrapped was further cleaned. We first filter the dataset to certain columns is relevant with our analysis and also check any empty cells in the dataset.  
Next we also remove any duplicates row based on the `title` and `selftext` column to avoid unnecessary noise.  

The following step, a new column `text` is created with the `title` and `selftext` value and the column is further check if any unnecessary string such as http/www address and html punctuations exist.

The EDA stage will be analysing the text using the Count Vectorizer and TF-IDF Vectorizer based on their unigram, bigram and trigram. Both vectorizer transforms our dataset into a structured dataset based on number of counts of the words as a whole which is more intepretable for the machine.  
The process include:
- Dropping Punctuation from the text
- Tokenising the text
- Lemmatizing the text
- Fit the model to the dataset
- Convert to pandas dataframe for plotting purposes

After removing the stopwords that occurs frequently from our dataset, we find that the Count Vectorizer and TF-IDF Vectorizer top words are similar.

The process of the modelling as follow:
1. Insert the transformer and estimator model to a pipeline
2. Set the pipe parameters
3. Instantiate GridSearchCV to find the best parameters to be fitted
4. Fit the GridSearchCV fitted model with the train and test data to calculate the accuracy
5. Find the features importance of each subreddit
6. Plot a confusion matrix and ROC Curve

We will do multiple combination of modelling to determine which one has the best indicator, here are the following combinations:
1. Count Vectorizer / Multinomial Naive Bayes
2. TF-IDF / Multinomial Naive Bayes
3. Count Vectorizer/ Binomial Naive Bayes
4. TF-IDF / Binomial Naive Bayes
5. Count Vectorizer / Logistic Regression
6. TF-IDF / Logistic Regression

Here is the comparison table of the accuracy score based on the model combinations
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky" colspan="2">Multinomial NB</th>
    <th class="tg-0pky" colspan="2">Bernoulli NB</th>
    <th class="tg-0pky" colspan="2">Logistic Regression</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">Count Vectorizer</td>
    <td class="tg-0lax">TF-IDF Vectorizer</td>
    <td class="tg-0lax">Count Vectorizer</td>
    <td class="tg-0lax">TF-IDF Vectorizer</td>
    <td class="tg-0lax">Count Vectorizer</td>
    <td class="tg-0lax">TF-IDF Vectorizer</td>
  </tr>
  <tr>
    <td class="tg-0pky">CV Score</td>
    <td class="tg-0pky">0.905294756</td>
    <td class="tg-0pky">0.898763067</td>
    <td class="tg-0pky">0.867411807</td>
    <td class="tg-0pky">0.809280194</td>
    <td class="tg-0pky">0.877207213</td>
    <td class="tg-0pky">0.892878585</td>  
  </tr>
  <tr>
    <td class="tg-0pky">Train Score</td>
    <td class="tg-0pky">0.967341607</td>
    <td class="tg-0pky">0.981058132</td>
    <td class="tg-0pky">0.935336381</td>
    <td class="tg-0pky">0.941868060</td>
    <td class="tg-0pky">0.980404964</td>
    <td class="tg-0pky">0.979751796</td>
  </tr>
  <tr>
    <td class="tg-0pky">Test Score</td>
    <td class="tg-0pky">0.913907285</td>
    <td class="tg-0pky">0.913907285</td>
    <td class="tg-0pky">0.850331126</td>
    <td class="tg-0pky">0.777483444</td>
    <td class="tg-0pky">0.872847682</td>
     <td class="tg-0pky">0.894039735</td>
  </tr>
</tbody>
</table>

The pipeline that we tested fair quite good as they have CV score higher than 0.80. It has an improvement of at least 30% from the baseline model accuracy.

As for this project, we consider the accuracy is the most important metrics that need to be analysed. Based on the CV score, we conclude that the Count Vectorizer with Multionomial Naive Bayes is the best model. It has the highest test score (it is the same as the TF-IDF Vecotrizer with Multinomial NB model). The other reason this is the best model, we can see the difference between the train score and test score is the lowest, this model overfitting is lesser than other models.  

The AUC score for every models remains high at around 0.95 but our model still has the best AUC.

The top 5 most important features for r/tea are:
- oolong
- matcha
- leaf
- pu
- herbal

The top 5 most important features for r/Coffee are:
- grinder
- espresso
- burr
- bean
- v60

The features are similar to what we found during the initial analysis in the EDA part.

---

### Conclusions:

**Conclusions :**
- Count Vectorizer + Mutinomial Naive Bayes Model is the best model for the classification
- Multinomial Naive Bayes are a better predictor than Logistic Regression in this project
- Best indicator for the tea is the type of tea such as oolong, matcha, pu erh and herbal
- As for the indicator for coffee, the best indicator is the method and the machine to make coffee.
- It may implies that coffee people enjoy to try different type of methods to get different/best flavours, meanwhile the tea lovers like to explore different type of teas.


**Recommendations :**  
1. Built the recommended system with top keywords and other related words
2. Maintain the inventory based on the trend provided from the analysis to have a greater supply chain effectiveness
3. Offer additional offerings that is related to the top keywords, such as matcha latte, RTD espresso

**Limitations :**
1. Misspelled words is not addressed thus it may fall out of context
2. Multi-words names of tea, beans, and their equipments are not addressed properly. Important terms that do not make sense as individual words such as pu erh or earl grey. 
3. Small stopword list, allow common words that has little relevant to tea and coffee to be included in the modelling
4. Text analysis only, does not include image and video analysis
5. Small datasets with only 1000s posts per subreddit
6. Reddit may have only posts mainly from US residents, other region remained untouched

**Future Works :**
1. Analysis of other social media platforms that may have bigger scope of global users
2. Analysis of forum of social media with other languages
3. Feature Engineering based on the parameter obtain from the scraped data and other type of data
4. Analysis of other form of data such as image and video
5. Scrape more data from the subreddit to be analysed further
---