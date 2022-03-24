# DS440-Project

Progress Report 5
Jie Zhu, Ben Jochem, Kashish Gujral
jzz5460, bmj5307, kmg6272 @psu.edu

Introduction

Our project investigates the efficacy of various machine learning algorithms at detecting fraud in three different areas of financial activity: Credit Card Transactions, Ethereum Network Transactions, and Insurance Claims. The goals of the project can be broken down into two categories: (1) Team Goals and (2) Individual Domain Specific Goals. Our primary Team Goal is to determine whether there is a machine learning algorithm that performs best in all three areas or whether algorithm performance is dependent on the area of financial activity. Our results offer direction for future, more rigorous investigations into machine learning techniques in different transactional contexts and benefit financial firms, individuals, and researchers that have an interest in machine learning applied to transactional data. We use the FLAML (A Fast Library for Automated Machine Learning & Tuning) python package for training, tuning, and evaluating the performance of various different classes of machine learning models. Individual Domain Specific Goals vary between the three team members. Each focuses on one of the three areas of financial activity and conducts an independent analysis on their own dataset. Figure 1 below is a diagram that details our project framework as a whole. In what follows we give an overview of each team member’s area of financial activity and Domain Specific Goals, summarize previous works related to our goals, describe our methodologies, detail our experiments, and discuss our results and conclusions.

Areas of Financial Activity

Credit Card Transactions

Due to the importance and scale of the credit card transactions in real life, Jie decides to do the Fraud detection about Credit Cards. The common fraud of credit cards is misusing others credit cards to pay intentionally or occasionally.  Jie finds that it is easy to find the effective database about this topic. Jie studied some algorithms during the previous semester (e.g. random forest, knn, decision tree). Therefore, she has an idea to implement these methods in this project. She searched dataset on Kaggle and found that there is a dataset about european credit cards accounts and can train the existing data to build a model and do testing data to predict the future transaction. Based on the common knowledge, Jie understands that the normal transaction usually happens during the day time, so it is worthwhile to notice that if there are some transactions during the night. Following this logic, we are able to detect the fraud in different ways. According to the professor's suggestion, Jie would like to take advantage of FLAML to work,  figure out the most effective algorithm and share the progress and result with the team as a whole. 

Jie uses a large amount of data to check whether the logistic regression is effective to detect credit card fraud in advance. The following goals will be progressed in the future work:

Classify credit card fraud by logistic regression so that we can get the training and testing data.

According to the fraud data unevenly distributed, it is important to figure out which metric is more suitable for this dataset (e.g. accuracy, precision, recall, and f1).

Try to visualize the result to show the difference between normal transaction and fraudulent transaction.

Data: https://www.kaggle.com/mlg-ulb/creditcardfraud

Insurance Claims

Insurance claim fraud is the deliberate deception against or by the company or agent in order to gain financially. These can be taken place in any point in the process of a transaction
These frauds include: (1) criminals who steal large amounts of money from businesses, (2) people who change service costs, or (3) normal people that cover their deductibles and file claims to make some extra money. Kashish has prior knowledge about Insurance Fraud as she worked on a similar dataset for a previous class. She understands the parameters and factors that play a role in this situation. She will continue to monitor any changes in the claims and work toward the team goal as a whole. The basic idea behind this category will be to work on the dataset and see what factor in insurance acts the most and have the biggest impacts when it comes to claims and frauds. She also wants to make a prediction model that can, for future purposes, predict if the transaction or claims filed might be a fraud. Furthermore, she will use data visualization and prepare charts/graphs for better visual analysis.

Data: https://www.kaggle.com/buntyshah/insurance-fraud-claims-detection/data
 
Ethereum Network Transactions

Given his interest in blockchains and cryptocurrencies, Ben investigates transactions on the Ethereum blockchain as his area of financial activity. These transactions are structurally similar to transactions using traditional payment systems such as credit cards but there are also some key distinctions. Ben explores whether these key distinctions impact the effectiveness of machine learning algorithms at detecting fraudulent transactions. Formally, he investigates the following three Domain Specific questions:

 Are the machine learning algorithms as a whole more or less effective at detecting fraudulent transactions on Ethereum as compared to on traditional payment systems?

Which features are the same and which differ between Ethereum transactions and transactions in traditional payment systems and which set of features are more relevant to detecting fraudulent transactions? 

Which machine learning algorithms perform better or worse and why might this be? What does this mean for the structure of fraud detection systems being built on Ethereum (i.e. how/if they will differ from traditional systems)

Data: https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network

Related Works

There has been much prior work exploring the performance of different machine learning algorithms for detecting financial fraud. Most previous research focuses on only one dataset and compares the performance of a few common algorithms using a few common metrics. This research is useful but it leaves a lot of questions on the table regarding why an algorithm may or may not work in a specific context. Our project seeks to answer these questions and is novel in that it compares multiple different machine learning algorithms across different datasets in different areas of financial activity. We hope that this will lead to a better understanding of algorithm performance in varying contexts and set the stage for more in-depth research in this area. What follows is a brief description of the prior work completed on the topic.

All previous work relevant to our problem investigates the performance of many machine learning algorithms on only one dataset (usually credit card transactions). Thus, our problem is unique and there is no prior work that can be used for direct comparison. We can, however, use the results from these prior works to compare the results from each of our areas of financial activities and make interesting comparisons. The common result amongst all of these prior works was that the KNN, Logistic, and DNN models are the most effective at predicting fraud. It should be noted that these results can likely only be extended to Jie’s work as she is also looking at credit card transactions. She will get the most use out of these works and will include in her analysis a comparison between them and her results. Ben and Kashish will look at the results from these works for their different datasets.

Methodology

Our Team Goal is to determine whether there is an algorithm that performs best in all three areas or whether algorithm performance is dependent on the area of financial activity so all team members must be consistent in their choices of algorithms and metrics for valid comparisons. We have chosen to use the FLAML python package to facilitate this consistency. The FLAML (A Fast Library for Automated Machine Learning & Tuning) package offers our team a standardized way to build and tune our models such that we can be sure that each of us followed the same process and that the comparison of our results is justified. Furthermore, the speed and efficiency of FLAML will allow us to streamline the model building process and focus on our primary question of interest which is a comparison of our results. The primary challenge in the early stages of our project was ensuring that each team member processed the data in the same way before modeling. This was achieved using principal component analysis. In the subsections that follow we describe the algorithms and metrics that we use in our project. Furthermore, each team member describes their methodologies for achieving their Domain Specific Goals. Figure 2 the methodology framework for achieving our Team goal.


Team Metrics

Our metrics are chosen based on simplicity and efficacy. We ensure that all team members have comparable results by simply using the metrics that are built into FLAML. A variety of metrics is used such that we are able to verify the efficacy of our models in different ways. It could be misleading if we only chose to use only a single metric such as accuracy because most observations in fraud detection datasets are negative (imbalanced) and by using accuracy we could get a high accuracy whilst not detecting any of the fraudulent transactions! The following four metrics provide us with a holistic view of the performance of our models: Precision, Recall, F1, and AUC. The combination of recall and precision is our main way by which we evaluate model performance

Team Models 

Like the metrics, our models were also chosen based on simplicity and efficacy. Based on our research of prior works and preliminary modeling tests, we use the following models: Xgboost, Random Forest, Support Vector Machines, and Ensemble methods.

Credit Card Transactions

Insurance Claims

Ethereum Network Transactions



Experiments

Detecting fraudulent financial activity is a classification task hence our choice to use supervised classification methods. Along with the chosen models and metrics listed in the preceding section, we plan on using a variety of different methods to create our models including standardization, normalization, PCA, and cross-validation. We have chosen a wide variety of methods such that we can combine them in different ways in each of our datasets thereby allowing us to better understand the context in which each method most effectively detects financial fraud. A detailed description of how each team member has begun to apply these methods to their dataset and the initial implementation results follow. 

Credit Card Transactions

I(Jie) work on analyzing my dataset this week. There are 31 features in my dataset including “Time”, “Amount”, and “V1-V28” (these are the features that already have been processed by PCA, therefore, we probably don’t know the exact name of these features). There is one feature called “Class” : Class = 0 means the particular transaction is normal; Class = 1 means the particular transaction is fraud. First, I filter the features to see whether all 30 features have a strong relationship with Class (whether or not it is a fraud transaction). This step helps future working by avoiding the unnecessary analysis. For each feature, I will make histograms to compare the fraud and normal transaction distribution. If the distribution (shape) of two histograms is similar, then the feature will be dropped. If there is a big (obvious) difference between two kinds of transaction in particular, then that feature will be kept. 

Examples: for the “Time” feature, the x-axis shows the time difference between each transaction and the first transaction. The y-axis shows the number of transactions.
From the screenshot below, we can generally conclude that the normal transaction follows the daily activity— during the day time, the number of transactions is large and reasonable;during the night, there are small numbers of transactions. However, for a fraud plot, the transaction seems to have nothing to do with time logic. Therefore, based on what I talked about in the previous paragraph, “Time” feature may be dropped. 



I did this process for remaining features, then I narrowed down the number of effective features from 30 to 17. 

Next week, I am going to work on the prediction model by logistic regression and randomforest. In the following weeks, our team will work on a combination of our three areas. We will try to make our process structure similarly to ensure that we are delivering the same idea and result. 

In week 8, I accomplish the feature flittering. I start to work on model building. Based on the knowledge I learned before, I use KNN and Random Deforest to train the model. However, maybe due to the dataset being large, the running time of the training model is too long. Therefore, I decided to generate a sample from the original dataset. Then use a random sample to train the model. There are 100 random samples. This method works in Randomforest, however, still does not work in KNN. Later on, I will try to use FLAML to work. I have already run some code with online samples, just trying to implement the code to my dataset. I was supposed to do the model training in week 8, however, the midterms slow down the working speed. Ben has already successfully run FLAML code with some classifications and he will share the idea with us. For the following weeks, the models for three region of fraud detection will be figured out. Then our group will meet together to compare the results and go back to the starting point. 

During the spring holiday, I plan to do some research about credit card fraud detection cases and other financial fraud detection areas. By comparing the existing method and analysis, I am going to see if there is anything that we can improve and if our models have already improved the existing ones.
 
Insurance Claims

This week, I started to work on the dataset that I finalized for the future final project. I looked over the 40 features that are included in the dataset (for instance, policy state, age, policy state, annual premium etc) I started to clean my data - which involved getting rid of some of the features which are not inputting much in the results I was looking for. This left me to work with 15 variables that had a strong connection with ‘fraud_detection’. Next, I made several density plots with different sets of variables in order to help me visualize how these features are related to each other and the ‘fraud detection’ and if I could find any similarities between the features and any patterns.

I used logistic regression for the next part. Basically, I made some models and ran them and then compared them with each other and hoped to get a suitable outcome with the features that I was working with. Below I added some snips from the first model that I started to work with.



    
After working with these random models, the last model( included policy_annual_premium + age + incident_severity)  that I was working with, I used Spike-Slab and got the following result. This will help me in the future prediction models.



The main goal for our team is to reach the final predictions and be able to compare each of these three business sectors and see what sectors have what fraud detected and how much the detection is. In the future weeks, we will have to work with similar models, metric systems in order to compare our models on a similar platform.

For this week’s progress, I continued to use the results from logistic regression models and compare results of the various models that I have come up with. As there are so many different features to take account with, with respect to fraud detection, I feel this is method is very time consuming. I also tried a new method called Naive Bayes and came up with some results with which I felt a little satisfied with. Below are the results with some graphs for easy understanding.







I feel confident with the pace of this project till now, but due to midterms these few days I wasn't able to give in much time with research and reading about it. Therefore I have decided that during spring break I will try to catch up with my other team members and come together on a single page. My goal for the next couple of weeks is to make FLAML run without any errors and try experimenting with different methods as KNN.



Ethereum Network Transactions

My initial work consisted primarily of data preprocessing and some initial model building. I preprocessed the data in such a way that I can easily apply all of the aforementioned methods (i.e. standardization, normalization, PCA) from the same base dataset. I was also able to run some initial models with FLAML and get a feel for how it works and how I may want to structure my analysis based on its outputs. Finally, throughout this process I also encountered some difficulties that our team will need to overcome in the coming weeks to achieve our project goal. 

My data preprocessing involved data cleaning and some basic data transformations. Given that my dataset was curated beforehand by Al-E’mari Et al, I have the benefit of relatively clean data. Many of the features, however, are only useful for data analysis and not helpful for predictive machine learning tasks, and needed to be removed. Namely, I removed the ‘hash’, ‘receipt_cumulative_gas_used’, ‘block_hash, and ‘block_number’ features. ‘hash’ refers to a Keccak-256 hash of the transaction which is random and therefore not useful for prediction. ‘receipt_cumulative_gas_used’ refers to the cumulative gas used in the block that the transaction was mined. This only relevant information in this feature for predictive purposes is contained in the ‘receipt_gas_used’ feature and was removed because of redundancy. ‘block_hash’ and ‘block_number’ are random and irrelevant respectively, thus they cannot provide any predictive value and were therefore removed. Next, I had to perform some basic data transformations on the remaining features to prepare them for application of our various methods. One transformation  was a mapping of the categorical input feature to a binary feature. This feature was initially a random hash of a message from the sender of a transaction to the receiver. Being a random hash the value of the content was irrelevant which is why I decided to transform it to a binary feature indicating whether a message was included in the transaction or not. A second transformation was changing the ‘value’ feature such that it is denominated in Ethereum’s native token Eth and not a smaller denomination called Wei. This was necessary because one Wei is such a small amount of value that the average transaction in the billions of Wei. This would have later caused problems during modeling and made results more difficult to interpret and convert to dollar terms. A final transformation was mapping a datetime feature to year, month, day, hour, minute and second features so that our algorithms could compute on the data. I only ended up keeping the month, day, hour and minute features as the others will likely be unhelpful in predicting fraudulent transactions for obvious reasons. These transformations can be found in the ‘Preprocessing.ipynb’ notebook in the linked Github repo.

After preprocessing my data I was able to implement some additional data processing and preliminary model building using FLAML. I opted to implement data standardization and fit a model using FLAML with the accuracy, AUC, and F1 metrics, and the xgboost, random forest, L1-Logistic, LGBM, L2-Logistic, and KNN models as a basic first approach. This process can be found at the end of the ‘Preprocessing.ipynb’ and beginning of the ‘Standardization_Model.ipynb’ notebooks. The basic FLAML model-building process was very simple, but I soon encountered some issues that our team will need to work out in the coming weeks. The first issue is that FLAML does not include the recall or precision metrics in the form that we would like to use them. We had decided that recall and precision are our primary evaluation metrics and therefore this will require us to code custom metrics. The second issue is that I continually received errors when trying to run the LGBM, L2-Logistic, and KNN models. It was not clear why these models were not running as the other models ran perfectly. Over the coming weeks we will need to resolve these issues. The good news, however, is that FLAML works as expected on our data and available output will allow us to achieve our end goal of comparing model performance across each area of financial activity. We will work as a team to resolve these common issues and continue with our project as it is currently planned.

Since the last project report Ben has continued to work towards implementing the custom metrics and debugging his code. He is in the process of finishing up implementing the last few models and will soon begin analyzing the results. Midterm exams have slowed down progress the last couple of weeks but overall he is still in a good place as compared to the timeline previously agreed upon. Over spring break he will finish up implementing his models and begin comparing his results with Jie and Kashish.

Github Repository: https://github.com/benjoch/DS440-Project

Discussion




Conclusion


References

D. Varmedja, M. Karanovic, S. Sladojevic, M. Arsenovic and A. Anderla, "Credit Card Fraud                          Detection - Machine Learning methods," 2019 18th International Symposium INFOTEH-JAHORINA (INFOTEH), 2019, pp. 1-5, doi: 10.1109/INFOTEH.2019.8717766.

Itoo, F., Meenakshi & Singh, S. “Comparison and analysis of logistic regression, Naïve Bayes and KNN machine learning algorithms for credit card fraud detection,” Int. j. inf. tecnol. 13, 1503–1511 (2021). doi: 10.1007/s41870-020-00430-y

J. O. Awoyemi, A. O. Adetunmbi and S. A. Oluwadare, "Credit card fraud detection using machine learning techniques: A comparative analysis," 2017 International Conference on Computing Networking and Informatics (ICCNI), 2017, pp. 1-9, doi: 10.1109/ICCNI.2017.8123782.


