# Ethereum-Fraud-Detection using Machine Learning


## Project Outline
In this project, fraudulent transactions are detected using machine learning techniques including 3 machine learning models, a Logistic Regression model, XGBoost model and a Multilayer Perceptron model. This project starts with Exploratory Data Analysis (EDA) in a separate notebook, Data Cleaning and Data Preprocessing, Model Implementation and then Evaluation. This is done on Jupyter Notebook.


## Introduction 

### Background
Ethereum is a decentralized global software platform powered by blockchain technology. It is most commonly known by investors for its native cryptocurrency, ether (ETH), and by developers for its use in blockchain and decentralized finance application development [1]. A blockchain is a distributed ledger with blocks containing records. These blocks are linked together like a chain using crypto hashes. Because of the structure of the blockchain and the fact that each block contains information about the previous blocks, the transactions in the blockchain are immutable and once a transaction or contract is recorded, it cannot be altered or else it would affect all subsequent blocks [2]. The blockchain is also based on principles of cryptography, decentralization and consensus, which ensure trust in transactions. Blockchain technology enables decentralization through the participation of members across a distributed network. There is no single point of failure, and a single user cannot change the record of transactions [3]. Even with the Blockchain secure characteristics, it is still vulnerable to fraud which is why fraud detection is needed. 

#### Problem Statement
Despite the transparency of blockchain data, the detection of fraudulent transactions remains a challenging task due to the complexity, volume, and evolving nature of fraudulent techniques. Traditional fraud detection methods often fall short in the blockchain context, where transactions are immutable and involve decentralized actors. Moreover, the inherent features of blockchain transactions such as the lack of centralized oversight, the variety of transaction types, and the global scale of operations complicate the identification of fraudulent behaviour.

#### Objective
This project aims to accurately identify fraudulent Ethereum transactions using Machine Learning.


## Implementation and Methodology

### Data Source
The dataset used in this project is from Kaggle: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset/data

#### About Dataset 
This dataset contains rows of known fraud and valid transactions made over Ethereum, a type of cryptocurrency. This dataset is imbalanced.

Content:

Here is a description of the rows of the dataset:

- Index: the index number of a row
- Address: the address of the ethereum account
- FLAG: whether the transaction is fraud or not
- Avg min between sent tnx: Average time between sent transactions for account in minutes
- Avg_min_between_received_tnx: Average time between received transactions for account in minutes
- Time_Diff_between_first_and_last(Mins): Time difference between the first and last transaction
- Sent_tnx: Total number of sent normal transactions
- Received_tnx: Total number of received normal transactions
- Number_of_Created_Contracts: Total Number of created contract transactions
- Unique_Received_From_Addresses: Total Unique addresses from which account received transactions
- Unique_Sent_To_Addresses20: Total Unique addresses from which account sent transactions
- Min_Value_Received: Minimum value in Ether ever received
- Max_Value_Received: Maximum value in Ether ever received
- Avg_Value_Received Average value in Ether ever received
- Min_Val_Sent: Minimum value of Ether ever sent
- Max_Val_Sent: Maximum value of Ether ever sent
- Avg_Val_Sent: Average value of Ether ever sent
- Min_Value_Sent_To_Contract: Minimum value of Ether sent to a contract
- Max_Value_Sent_To_Contract: Maximum value of Ether sent to a contract
- Avg_Value_Sent_To_Contract: Average value of Ether sent to contracts
- Total_Transactions(Including_Tnx_to_Create_Contract): Total number of transactions
- Total_Ether_Sent:Total Ether sent for account address
- Total_Ether_Received: Total Ether received for account address
- Total_Ether_Sent_Contracts: Total Ether sent to Contract addresses
- Total_Ether_Balance: Total Ether Balance following enacted transactions
- Total_ERC20_Tnxs: Total number of ERC20 token transfer transactions
- ERC20_Total_Ether_Received: Total ERC20 token received transactions in Ether
- ERC20_Total_Ether_Sent: Total ERC20 token sent transactions in Ether
- ERC20_Total_Ether_Sent_Contract: Total ERC20 token transfer to other contracts in Ether
- ERC20_Uniq_Sent_Addr: Number of ERC20 token transactions sent to Unique account addresses
- ERC20_Uniq_Rec_Addr: Number of ERC20 token transactions received from Unique addresses
- ERC20_Uniq_Rec_Contract_Addr: Number of ERC20 token transactions received from Unique contract addresses
- ERC20_Avg_Time_Between_Sent_Tnx: Average time between ERC20 token sent transactions in minutes
- ERC20_Avg_Time_Between_Rec_Tnx: Average time between ERC20 token received transactions in minutes
- ERC20_Avg_Time_Between_Contract_Tnx: Average time ERC20 token between sent token transactions
- ERC20_Min_Val_Rec: Minimum value in Ether received from ERC20 token transactions for account
- ERC20_Max_Val_Rec: Maximum value in Ether received from ERC20 token transactions for account
- ERC20_Avg_Val_Rec: Average value in Ether received from ERC20 token transactions for account
- ERC20_Min_Val_Sent: Minimum value in Ether sent from ERC20 token transactions for account
- ERC20_Max_Val_Sent: Maximum value in Ether sent from ERC20 token transactions for account
- ERC20_Avg_Val_Sent: Average value in Ether sent from ERC20 token transactions for account
- ERC20_Uniq_Sent_Token_Name: Number of Unique ERC20 tokens transferred
- ERC20_Uniq_Rec_Token_Name: Number of Unique ERC20 tokens received
- ERC20_Most_Sent_Token_Type: Most sent token for account via ERC20 transaction
- ERC20_Most_Rec_Token_Type: Most received token for account via ERC20 transactions

### Exploratory Data Analysis 

EDA was carried out using Matplotlib, Plotly and Seaborn libraries.  From the EDA, I can see that the data is significantly imbalanced with my non fraudulent transactions. From analysis I can also see that for fraudulent transactions, the average time difference between transactions is shorter. For Crypto types, the type with the most fraud was blank which could indicate methods of fraud or flaw in the data. The features overall included some that were highly correlated with each other however not with the FLAG column itself.

### Data Preprocessing

Several preprocessing techniques were used in this project.

#### Feature Selection

First irrelevant columns such as 'Index', 'Address', 'Unnamed:_0' where dropped. The columns containing only 0 values were dropped as they add noise to the data and are not valuable. Net highly correlated values are dropped. This is done to reduce the chances of overfitting, reduce the complexity of the model as these columns are similar, having too many features makes the dataset more complex and prevents the model from multicollinearity.

##### Handling Missing Values

Missing values are dropped using the most common value. For all the columns that needed to be filled, the columns were skewed with the mode being significantly higher in count so using the mode would not affect the data as much as using the mode which would create a spike elsewhere. 

#### Encoding Categorical Features

One-Hot-Encoding was used to encode two categorial columns, 'ERC20_most_sent_token_type' and 'ERC20_most_rec_token_type'. One-Hot-Encoding is used as it prevents the model from assuming that there is any ordinal relationship between categories [5]. This was done using the Pandas get_dummies method [4] for simplicity and the result is a dataframe so there is no need to concatenate it later. 

#### Rebalancing 

Because the data is imbalanced, rebalancing is conducted using the SMOTE method. SMOTE (Synthetic Minority Over-sampling Technique) is a popular method used to address class imbalance in datasets, particularly in classification problems where one class is significantly underrepresented compared to the others like in this dataset. This was done using the Imbalanced-Learn library [6]. 

#### Normalisation 

Since the features are all in different ranges, context and since the range between particularly large, this can affect the model's performance. Dimensionality Reduction is also carried out later so scaling the features is necessary. Here the Standard Scaler from the SciKit-Learn library is used to scale the dataset [7].

#### Dimensionality Reduction

After using One-Hot-Encoding, the dataset has over 800 columns which can introduce new issues with the high complexity. To tackle this, Principal Component Analysis (PCA) is used to reduce the dimensions of the dataset. PCA is implemented from the SciKit-Learn library. First the optimal dimension is calculated using 'pca.explained_variance_ratio' which is 412 and then PCA reduced the dimensions to the optimal number of dimensions [8].

### Model Implementation 

For this project, I wanted to test and compare three models with different complexity to see if it would affect the performance and accuracy. Here I implemented a Logistic Regression, XGBoost Classifier and Multilayer Perceptron model from SciKit-Learn [9], XGBoost[10] and Tensorflow librarys imported into Jupyter Notebook.   

For the XGBoost and Multilayer Perceptron model, I used Hyperparameter Tuning to see if there was a best model. I used RandomSearchCV from SciKit-Learn ad GridSearchCV was too time consuming and computationally expensive, and I do not have the resources for it. For both models the hyperparameter tuning was not optimal which might be due to the search being random. 

The Multilayer Perceptron model has a few adjustments whiles testing the model. I had to test and adjust the number of layers and Neurons to see their effects on the performance whiles not be too computationally consuming [12].

To evaluate the models I used an Accuracy score as well as the 'classification_report' method to see the Recall, Precision and F1 Score. A Confusion Matrix was also used to compare the True Positives, True Negatives and the False Positives and False Negatives. 

## Results 
Overall the Multilayer Perceptron model performed the best with an accuracy of 0.78293 whiles the Logistic Regression model performed the poorest with an accuracy of 0.638672. This was expected because of their respected complexities. 

## Challenges and Limitations 

Challenges to this project was mainly in the data preprocessing stage as the data needed to go through several preprocessing stages. This is the first time I have used one hot encoding for my categorical variables which also led on to me using dimensionality reduction for the first time. At first, I did not do enough data preprocessing which as a result gave me a poor performing logistic regression model and XGBoost model with their accuracy being under 0.4. This made me acknowledge the importance of preparing the data to a good standard for modelling.

A limitation to this project is the Hyperparameter Tuning method RandomSearchCV which meant that the tuning was not optimal as it could not search through all of the options. With more resources I would have liked to use GridSearchCV however running time was too time consuming. 


## Conclusion and Reflection

This is my first independent Classification project which is on Ethereum Fraud Detection. I wanted to get some experience on doing a classification problem as at this time, I have only done regression problems. I wanted to include some hyperparameter tuning to gain more confidence in including tuning in my projects and I also wanted to include a neural network model as I have little experience with neural networks, and I find them challenging.  

Overall, there were many learning outcomes to this project and I am satisfied with the results. For future work I would like to include more advanced hyperparameter tuning methods for all my models and also further process the data with more advanced methods.



## References 

[1] Frankenfield, J. (2021). Ethereum. [online] Investopedia. Available at: https://www.investopedia.com/terms/e/ethereum.asp.  
[2] Wikipedia (2019). Blockchain. [online] Wikipedia. Available at: https://en.wikipedia.org/wiki/Blockchain.  
‌[3] www.ibm.com. (n.d.). What is Blockchain Security? | IBM. [online] Available at: https://www.ibm.com/topics/blockchain-security#:~:text=Each%20new%20block%20connects%20to.  
[4] Scikit-learn (2019). sklearn.preprocessing.OneHotEncoder — scikit-learn 0.22 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html.  
[5] Brownlee, J. (2017). Why One-Hot Encode Data in Machine Learning? [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/.  
[6] imbalanced-learn.org. (n.d.). SMOTE — Version 0.9.0. [online] Available at: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html.  
[7] Scikit-Learn (2019). sklearn.preprocessing.StandardScaler — scikit-learn 0.21.2 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html.   
[8] scikit-learn (2009). sklearn.decomposition.PCA — scikit-learn 0.20.3 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.   
[9] scikit-learn (2014). sklearn.linear_model.LogisticRegression — scikit-learn 0.21.2 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.  
[10] xgboost developers (2022). XGBoost Documentation — xgboost 1.5.1 documentation. [online] xgboost.readthedocs.io. Available at: https://xgboost.readthedocs.io/en/stable/.  
[11] Educative. (n.d.). Educative Answers - Trusted Answers to Developer Questions. [online] Available at: https://www.educative.io/answers/classification-using-xgboost-in-python.
[12] https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/  
‌[13] Dmitry Noranovich (2021). Binary classification with TensorFlow 2 - Dmitry Noranovich - Medium. [online] Medium. Available at: https://javaeeeee.medium.com/binary-classification-with-tensorflow-2-bc70f0cc7035 [Accessed 29 Aug. 2024].

‌

‌

‌

