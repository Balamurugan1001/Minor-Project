**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY** 

**Classification Of Tamil News Articles By Naive Bayes Model Using Natural Language Processing**  

**A MINOR PROJECT REPORT** 

**Submitted by Balamurugan S - RA1811003020004** 

**Isvariyashree H - RA1811003020036                                      Senthurgokul B - RA1811003020021** 

**Under the guidance of** 

**Mrs. Saraswathi.E** 

**(Assistant Professor, Department of Computer Science and Engineering)** 

***in partial fulfillment for the award of the degree of*** 

**BACHELOR OF TECHNOLOGY** 

***in*** 

**COMPUTER SCIENCE AND ENGINEERING** 

**of** 

**FACULTY OF ENGINEERING AND TECHNOLOGY** 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.001.png)

**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY**

**RAMAPURAM CAMPUS, CHENNAI -600089**

**NOV 2021** 

**(Deemed to be University U/S 3 of UGC Act, 1956)** 

**BONAFIDE CERTIFICATE** 

Certified that this project report titled **“CLASSIFICATION OF TAMIL NEWS ARTICLES BY NAÏVE BAYES MODEL USING NATURAL LANGUAGE PROCESSING”**is the bonafide work **of Balamurugan.S RA1811003020004,Isvariyashree.H RA1811003020036 &Senthurgokul.B RA1811003020021** who carried out the project work under my supervision. Certified further, that to the best of my knowledge the work reported herein does not form any other project report or dissertation on the basis of which a degree or award was conferred on an occasion on this or any other candidate.

PAGE \\* romanii 
**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY** 

SIGNATURE 

**Mrs.Sarawathi.E** 

**Assistant Professor,** 

Computer Science and Engineering, 

SRM Institute of Science and Technology, Ramapuram Campus, Chennai. 

SIGNATURE 

**Dr. K.RAJA,M.E., Ph.D.,** 

**Professor and Head** 

Computer Science and Engineering, 

SRM Institute of Science and Technology, Ramapuram Campus, Chennai. 

PAGE \\* roman 
**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY** 

Submitted for the project viva-voce  held on ................................. at SRM Institute of Science and Technology , Ramapuram Campus, Chennai -600089. 

**INTERNAL EXAMINER-I      INTERNAL EXAMINER-II** 

PAGE \\* romaniii 
**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY**  

`   `**RAMAPURAM, CHENNAI - 89 DECLARATION** 

We hereby declare that the entire work contained in this project report titled “**CLASSIFICATION  OF  TAMIL  NEWS  ARTICLES  BY  NAIVE BAYES MODEL USING NATURAL LANGUAGE PROCESSING”**has been carried out by Balamurugan.S RA1811003020036,Isvariyashree.H RA1811003020036 & Senthur gokul.B RA1811003020021 at SRM Institute of Science and Technology, Ramapuram  Campus,  Chennai-  600089,  under  the  guidance  of **Mrs.SARASWATHI.E,  Assistant  Professor**,  Department  of  Computer Science and Engineering. 

**Place: Chennai                                                                           BALAMURUGAN.S** 

`                                                                                                                      `**ISVARIYASHREE.H                                                                                                                        SENTHUR GOKUL.B** 

**Date:**  

PAGE \\* romaniii 

**ABSTRACT** 

`          `Machine learning has created a drastic impact in every sector that has integrated it into business processes such as education, healthcare, banking services, etc. The current development of Machine Learning algorithms helps to attain effective Tamil document classification. Automatic text classification aims to allocate fixed class labels to unclassified text documents. NLP problems are unclear for languages other than English. The problems may be named as Entity Extraction, OCR or classification. So in this project we are going to use Naive Bayes model to categorize the Tamil articles in an efficient way. With the help of this Naïve bayes technique, the dataset and the trained model we can achieve the desired output within the stipulated time. So that with the help of the trained model, we can achieve the desired accuracy at a higher level. 

PAGE \\* romaniv 

**APPENDIX 3** 

**TABLE OF CONTENTS** 

PAGE \\* romanv 

**CHAPTER NO.               TITLE** 

**ABSTRACT** 

**LIST OF FIGURES LIST OF TABLES** 

**1  INTRODUCTION 2  LITERATURE** 

**SURVEY** 

**PAGE NO.** 

`            `**iv** 

**vii**   

**ix** 

**1 6** 

PAGE \\* roman 

**3  SYSTEM  16** 

**DESIGN**                                    

1. **INTRODUCTION** 
1. **SYSTEM ARCHITECTURE** 
1. **SYSTEM REQUIREMENTS** 
1. **SUMMARY** 

**4                MODULE  23** 

`             `**DESCRIPTION** 

1. **INTRODUCTION** 
1. **SCRAPPING THE DATA** 
1. **TEXT PRE- PROCESSING** 
1. **TRAINING AND TESTING DATA AND HYPERPARAMETER TUNING** 

PAGE \\* romanvi 
5. **BAG OF** 

**WORDS, TRAINING THE MODEL AND INFERENCE** 

**4.6 SUMMARY** 

`    `**5  SYSTEM  42** 

**IMPLEMENTATION** 

**6  TESTING  58 7  RESULT ANALYSIS  62 8  CONCLUSION AND  67** 

**FUTURE WORKS** 

PAGE \\* romanvii 

**LIST OF FIGURES** 



|**FIGURE NO** |**FIGURE NAME** |**PAGE NO** |
| - | - | - |
|1 |System architecture flow diagram |16 |
|2 |Data  |25 |
|3 |Scraping the data |26 |
|4 |Text pre-processing |27 |
|5 |Tokenization |28 |
|6 |Lemmatization |29 |
|7 |Stemming  |30 |
|8 |Stop words removal |32 |
|9 |Vectorization |33 |
|10 |Label encoding |34 |
|11 |Flow chart of training and testing data |35 |
|12 |Flow chart for training the model |38 |
|13 |Naive bayes formula |39 |
|14 |Importing the required Libraries |52 |
|15 |Importing the Dataset |52 |
|16 |Visualizing the categories and Splitting of Datasets |53 |
|17 |Data Cleaning |54 |
|18 |Cleaning of X,Y Train and Test Data set |55 |
|19 |Training  the  dataset  using Naïve  bayes  and  Tuning using  Hyper  parameter tuning |55 |

PAGE \\* romanix 



|20 |Precision, Recall, f1-score and support |56 |
| - | :- | - |
|21 |Confusion matrix |56 |
|22 |Inferencing |57 |
|23 |Data cleaning |59 |
|24 |Transformation of vectors |60 |
|25 |Number of categories in the dataset |61 |
|26 |F1 Score |63 |
|27 |Precision |63 |
|28 |Classification report of Naive Bayes|64 |
|29 |Confusion Matrix of Naive Bayes Classifier|65 |
**LIST OF TABLES** 



|**TABLE NO** |**TABLE NAME** |**PAGE NO** |
| - | - | - |
|1. |Accuracy of train, test and cross validation datasets |66 |

PAGE \\* romanxi 
`                                                       `**CHAPTER PAGE1** 

**1.1] OVERVIEW** 

Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as “training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks. In this machine learning project, we will recognize handwritten characters, by modeling a neural network. The Web is fast evolving into a platform for mass participation in content creation and consumption, with a rising number of people resorting to online news sources for daily updates. Around the same time as the internet became widely available in Bangladesh, a Tamil online newspaper began to appear. Every day, a large number of Tamil news stories are generated by the various news sites that exist on the Internet, and the rate continues to rise tremendously. An online newspaper can take many different formats. The electronic edition of a printed newspaper is one example.The online edition can be read in the same way as a paper edition; there is no categorization, neither in terms of content nor in terms of presentation. The use of Indian regional languages such as Tamil, Hindi, Telugu, and others on the internet is growing by the day. It’s a difficult process to classify such data by topic. Because the learning process for regional languages like Tamil is different from that of English. The task of classifying Tamil data by topic is a text classification task. Many applications, such as web searching, information filtering, language recognition, readability rating, and sentiment analysis, already use text categorization. The dataset development and evaluation of pretrained word embeddings of Tamil words are two of our primary contributions to this work. A news website is another type of online newspaper that allows users to browse through menus sorted by subject areas and sub-categories. The user is presumed to read the news via a computer screen while connecting to a specific news provider over the Internet in most of the above types of online newspapers. These services, however, 

PAGE1 

may not be enough for many readers and reading contexts. Many newspaper readers enjoy reading and analysing news from numerous sources. Readers are frequently only interested in news stories related to their areas of interest. As a result, visitors must sift through several news stories in order to find the ones that are of interest to them. A user interested in sports news, for example, must read all of the news stories from numerous news sites and spend time analysing news from multiple sources, which is time consuming. As a result, a reader would prefer a system that collects news articles from many sources and makes them available at all times and from any location, including via a mobile reading device. A reader would want to visit a one-of- kind newspaper that has stories from a variety of favourite sources, sorted and presented in the order that best suits her interests and reading habits. Natural language processing (NLP) is a branch of artificial intelligence (AI) that studies human-computer interactions using natural languages, such as word, phrase, and sentence meanings, as well as syntactic and semantic processing.To interpret and reason about a text, early NLP research relied on rule-based approaches. These rules are manually created by experts for a variety of NLP activities.Naive bayes is a classification strategy based on Bayes’ Theorem and the premise of predictor independence. The Naive Bayes model is simple to construct and is particularly useful for huge data sets. Naive Bayes is acknowledged to outperform even the most advanced classification systems, owing to its simplicity.Characterization and forecasting are two important activities for data analysis and model induction with the purpose of depicting important groups of data, obtaining them, and forecasting their future behaviour.. 

Natural language processing (NLP) is a subfield of artificial intelligence (AI) that can study human and computer interactions through natural languages, such as the meaning of words, phrases, sentences, and syntactic and semantic processing. In early, NLP research used rule-based methods to understand and reason a text. Experts manually create these rules for various NLP tasks. Naive bayes is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. The Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods. Text organisation is, in most cases, a two- step process. The construction of the model is the first stage, followed by the application of the model and forecasting based on historical data. The order of text 

PAGE3 

archives entails grouping these records and messages into groups with basic characteristics so that these reports can be used later based on this common feature.News classification is the process of classifying news documents into predetermined groups based on their content.For news classification in many languages, a number of statistical and machine learning algorithms have been developed. The exponential rise of the Internet in recent years has resulted in a vast quantity of electronic materials in a variety of regional languages other than English. The large number of Tamil documents generated by news, blogs, eBooks, and entertainment, automated classification of Tamil documents is required. This study focuses on the creation of deep learning (DL) models for Tamil document classification because automated Tamil document classification is not well understood. For Tamil documents, this work introduces an ensemble of feature selection using DL- based classification models.TFIDF, Chi Squared (CS), and Extra Tree (ET) Classifier models are utilised For classification, the suggested method employs deep neural network (DNN) and convolutional neural network (CNN) models.Gradient Boosting is a strategy used by Rajkumar et al [2] to create an additive predictive model by integrating different weak predictors, often Decision Trees. Gradient Boosting Trees can be used for classification as well as regression. To further understand how GBT works, we’ll use a binary outcome model. Sentiment Analysis (SA) is an application of Natural Language Processing (NLP) to extract the sentiments represented in the text.. We tested five ways to perform SA in this paper: a Lexicon-based approach, a Supervised Machine Learning-based approach, a Hybrid approach, a K-means with Bag of Word (BoW) approach, and a K-mode with BoW approach. To anticipate the optimal strategy to perform SA in Tamil texts, we tested various approaches using five corpora with different feature representation strategies. Named Entity Recognition (NER) is a subsequence of words in a text that attempts to recognise and classify entities into predefined categories such as the person’s name, organisation, and location. Because many Information Extraction (IE) relations are related using Named Entities, the influence on NER is significant (NEs). Because of its simplicity and effectiveness, the Naive Bayes (NaiveBayes) probabilistic classifier is frequently used in text classification applications and experiments. Its basic idea is to estimate the probabilities of categories given a document using the joint probabilities of words and categories.  

**1.2] PROBLEM STATEMENT**  

` `The Problem statement is to categorize the tamil news articles using Naive Bayes Classifier .The challenge is aimed at making use of machine learning and NLP techniques in categorizing the Tamil news articles which is available online into various different class labels such as sports, technology, cinema and etc

**1.3]OBJECTIVE** 

The main objective of this project is to Categorize the Tamil news articles available by Naive Bayes model using Natural Language Processing which comes under the domain called Machine learning. 

- Selecting and building a powerful model for categorizing the Tamil news articles. 
- To achieve the desired output within the stipulated time with the help of this Naïve bayes technique, the dataset taken and the trained model. The assumption of word independence is 

the naive aspect of such a paradigm.  

Text classification is a machine learning technique for classifying unstructured text into a set of predefined categories.Text classifiers can organise, classify, and organise nearly any sort of text, including documents, medical studies, and files, as well as text from the internet. For example, new articles can be classified by subjects; support tickets can be selected; chat conversations can be classified  by  language;  brand  mentions  can  be  classified  by  sentiment;  and  so  on.  Text categorization is a fundamental problem in natural language processing with numerous applications such as sentiment analysis, subject tagging, spam detection, and intent detection. 

**1.4] ORGANIZATION OF THE REPORT** 

The project mainly focuses in categorizing the Tamil news articles which is available online into various different class labels such as sports, technology, cinema and etc by making use of naive bayes classifier, Machine learning and NLP techniques.

**CHAPTER 1** 

This chapter deals with the overview of the project, problem statement of the project and objective of the project. 

**CHAPTER 2** 

This chapter contains literature survey which describes the already existing system and their advantages and disadvantages. 

**CHAPTER 3** 

`                    `This  chapter  includes  the  system  specifications  with  a  detailed  description  of architecture of the system. 

**CHAPTER 4** 

`                    `This chapter deals with the module description and therefore the flow of the process is also explained. 

**CHAPTER 5** 

This chapter deals with system implementation in which overview of the platform, code snippets and screenshot of outputs are depicted. 

**CHAPTER 6** 

`                   `This chapter entirely deals about the testing on the platforms used. 

**CHAPTER 7** 

This chapter deals with the result analysis of the project and therefore the outcome of 

the project. 

**CHAPTER 8** 

This chapter deals with the conclusion and future enhancement of the project. 

PAGE6 
**CHAPTER PAGE7** 

**2.] LITERATURE SURVEY 2.1] INTRODUCTION:** 

` `A literature review establishes familiarity with and understanding of current research in a particular field before carrying out a new investigation. Conducting a literature review has enabled us to find out what research has already been done and to identify what is unknown within our topic.Basically the research that has been done here for our project is related to classification of Tamil news articles by using NLP. Here we have taken ten different research papers for the literature review.The major concepts such as stemming, decision tree algorithm,TF- IDF, gradient boosting tree and document clustering existed in these ten research papers. 

**2.2] EXISTING SYSTEM** 

**2.1] A survey on N. Rajkumar, T. S. Subashini, K. Rajan and V. Ramalingam, An Ensemble of Feature Selection with Deep Learning based Automated Tamil Document Classification Models, International Journal of Electrical Engineering and Technology, 11(9) in 2020.** 

In recent times, the exponential growth of the Internet has resulted in an enormous number of electronic documents in several regional languages apart from English. Numerous documents in Tamil language are being generated from news, blogs, eBooks, and entertainment, the automated classification of Tamil documents is needed. Since the automated Tamil document classification is not discovered proficiently, this study focuses on the development of deep learning (DL) models for Tamil document classification. This paper introduces an ensemble of feature selection with DL based classification models for Tamil documents. The presented model primarily involves preprocessing to remove the unwanted data and improve the data quality to a certain extent. Besides, term frequency–inverse document frequency (TF-IDF) approaches were used to extract the features from the Tamil documents. In addition, two feature selection (FS) techniques namely Chi Squared(CS) and Extra Tree (ET) Classifier models were employed. The proposed method also uses deep neural network (DNN) and convolutional neural network (CNN) models for classification purposes. A detailed experimentation analysis takes place using a Tamil document dataset gathered by their own. The experimental values showcased that the ETFS-CNN model has obtained an effective classification outcome with the maximum accuracy of 90%, 

PAGE7 

precision of 90.57%, recall of 90%, and F-score of 89.89%.  Concepts used: Stemming, Decision Tree Algorithm 

Merits: TF-IDF Approach was useful for the extraction of features Demerits: Accuracy was not up to the level.  

**2.2] A survey on An Efficient Feature Extraction with Subset Selection Model using Machine Learning Techniques for Tamil Documents Classification by N. Rajkumar, T. S. Subashini, K. Rajan and V. Ramalingam in 2020.** 

As Tamil Text data in digital format both in online and offline mode is growing significantly nowadays, management and retrieval of the documents is a tedious process. Automatic text classification aims to allocate fixed class labels to unclassified text documents. Many natural language processing (NLP) techniques were extremely dependent on the automatic classification of Tamil Text documents. The current development of machine learning (ML) algorithms helped to attain effective Tamil document classification. In this view, this paper introduces an automated Tamil document classification technique using ML models. The presented model involves different processes such as preprocessing, feature extraction, feature selection, and classification. The proposed model uses the term frequency-inverse document frequency (TF-IDF) approach for the feature extraction process. Besides, the Chi-square test were employed to select an optimal set of features. At last, three ML models such as random forest (RF), decision tree (DT), and gradient boosting tree (GBT) were applied to determine the class labels of the Tamil documents. To assess the performance of the presented model, a set of simulations takes place on a Tamil document dataset collected on our own. The experimental values ensured the effective classifier results of the presented model over the compared methods.This paper has developed an effective automated Tamil document classification technique using ML models. The input Tamil documents initially undergo preprocessing to discard the unwanted data and perform necessary operations to make it compatible with the further processes. Followed by, the TF-IDF technique were employed for the extraction of useful features, and the feature subset were selected by the Chi-square technique. At last, the features were fed into the ML models for classifying the documents effectively. To validate the performance of the presented model, a series of simulations took place using a dataset collected 

PAGE8 

on their own.From the experimental values, it is ensured that the GBT model has reached an effective classification outcome with the maximum accuracy of 85.10%, precision of 87.01%, recall of 85.10%, and F1-score of 85.52%. 

Concepts used: TF-IDF, Decision Tree, Gradient Boosting Tree 

Merits: GBT Model has reached an effective classification outcome 

Demerits: RL and DL Does not achieve much accuracy compared to other classifiers. 

**2.3] Sentiment Analysis in Tamil Texts: A Study on Machine Learning Techniques and Feature Representation  published by S. Thavareesan and S. Mahesan in 2019.** 

Sentiment Analysis is the process of identifying and categorising the sentiments expressed in a text into positive or negative. The words which carry the sentiments are the keys in sentiment prediction. The SentiWordNet is the sentiment lexicon used to determine the sentiment of texts. There were a huge number of sentiment terms that were not in the SentiWordNet limit the performance of Sentiment Analysis. Gathering and grouping such sentiment words manually is a tedious task. In this paper they have proposed a sentiment lexicon expansion method using Word2vec and fastText word embeddings along with rule-based Sentiment Analysis method.Concepts Used: K-Means,Bag of Words and NLP. The main aim of this paper was to propose a rule-based Sentiment Analysis method with an efficient lexicon to predict the sentiments expressed in the Tamil text into positive or negative. Sentiment words are the key influences to determine the sentiments expressed in the text. The effectiveness of the lexicon based Sentiment Analysis method entirely depends on the sentiment lexicon. This paper  proposed a method to enlarge the human-annotated gold standard sentiment lexicon by using Word2vec and fastText word embedding and the enlarged lexicon which were  used to build our Sentiment Analysis method.We can conclude that, the Sentiment Analysis on Tamil text using UJ\_Lex\_Pos and UJ\_Lex\_Neg lexicons, negation and conjunctions gives a satisfying results with 88 ± 0.14% accuracy, 0.79 precision and 0.80 recall. In the future, this paper is planning to assign polarity values such as strong and weak to the words in this expanded lexicon with the help of linguistic experts. Concept:Sentiment analysis, Tamil, lexicon, conjunction and grammar rule 

Merits: It has an effective classification outcome 

Demerits: Various Mathematical formulas are used and the data training is not accurate. **2.4] Automated Named Entity Recognition for Tamil Documents by R.Srinivasan in 2019** 

Named Entity Recognition (NER) is a subsequence of words in a document that seeks to detect and 

classify entities into predefined categories such as name of the person, organization and location respectively. The impact on NER is high because a lot of Information Extraction (IE) relations are associated using Named Entities (NEs). This paper presents a pioneering method for extraction of NEs for Tamil using Supervised Learning. This hybrid framework makes use of features that are extracted based on the speciality of the Tamil language NEs. The evaluation has been done by using 1028 number of documents which comprises the standard FIRE corpus and an F-measure of 83.54% has been achieved. A performance comparison with one of the state-of-the-art Tamil NE systems has been done and the proposed methodology has achieved better accuracy.In this paper, they have proposed a new technique for named entity recognition for Tamil language. The proposed methodology differs from the state of art techniques by grouping the features and tapping the best out of the features by hierarchical ordering of features. This has increased the efficiency of the proposed technique. The bootstrapping technique has aided in increasing the coverage of features. Since the features are perfectly grouped into three classes, the prediction of the Entity class using the Naïve Bayes algorithm has increased. Since language like Tamil lack a benchmark data set, the proposed technique would be a pointer in creating such data sets in future. Concepts used: Named Entity Recognition, Information Extraction 

Merits:  Naïve bayes algorithm has contributed in increasing the coverage of feature extraction Demerits: Focuses on only Five entities. 

**2.5]Tamil News Clustering Using Word Embeddings by M.S. Faathima Fayaza and Surangika Ranathunga in 2020** 

News aggregators support the readers to view news from multiple news providers via a single point. At the moment, the only news aggregator that supports Tamil news is Google news, which has some noticeable shortages. In this study, Term Frequency–Inverse Document Frequency and word embedding (fastText) document representation techniques were experimented with one pass and affinity propagation clustering algorithms to news title, as well as title and body in order to implement a news aggregator for the Tamil language. For this study they have collected data from nine different news providers. When fastText was applied with one pass algorithm to news title and body, it managed to beat other approaches to achieve an average pairwise F-score of 81% with respect to manual clustering. Also, they were able to create a Tamil fastText word embedding model using more than 21 million words. This paper presented a Tamil news clustering system consisting of three modules, namely the Tamil news article data collection module, data representation module and Tamil news article clustering module. Data representation module was implemented using Term Frequency–Inverse Document Frequency (TF-IDF) and fastText document representations. One-pass and affinity propagation algorithms were used for clustering. Further in this study they have used title alone, and title and the body of the news to experiment the different approaches. They achieved the best result when they used the title and body of the news to cluster.They have also created a fastText word embedding model with 21 million words. From their study they achieved the best result by using fastText with one pass clustering algorithm. On the whole they achieved 81% accuracy with a standard deviation of 0.047 across 10 datasets. From their study they  have observed that fastText is able to identify words written in different styles by different publishers. Also, fastText can handle inflected words, in contrast to TF-IDF. One pass clustering algorithm outperforms the affinity propagation algorithm with both document representation approaches. It may be due to having a high number of single article clusters in the dataset. There were a number of possible ways to improve the performance of the current system. 

Concepts used: Document clustering, Tamil, word embedding, Term Frequency–Inverse Document Frequency, affinity propagation clustering, one pass algorithm 

Merits: Fast text word embedding model was created with 21 million words 

Demerits: No stop words removal approach were used 

**2.6] HUB@DravidianLangTech-EACL2021: Identify and Classify Offensive Text in Multilingual Code Mixing in Social Media byBo Huang and Yang Bai in 2020.** 

This paper introduces the system description of the HUB team participating in DravidianLangTech - EACL2021: Offensive Language Identification in Dravidian Languages. The theme of this shared task is the detection of offensive content in social media. Among the known tasks related to offensive speech detection, this is the first task to detect offensive comments posted in social media comments in the Dravidian language. The task organizer team provided them with the code-mixing task data set mainly composed of three different languages: Malayalam, Kannada, and Tamil. The tasks on the code mixed data in these three different languages can be seen as three different comment/post-level classification tasks. The task on the Malayalam data set is a five-category classification task, and the Kannada and Tamil language data sets are two six-category classification tasks. Based on their  analysis of the task description and task data set, they have chosen to use the multilingual BERT model to complete the task. In this paper, they have discussed about fine-tuning methods, models, experiments, and results.On the three different language data sets provided by the task organizer, they have combined the Tf- Idf algorithm and the output of the multilingual BERT 208 model and also they have introduced the CNN block as a shared layer. Experimental results proved that the conjecture is feasible.On the Kannada and Tami data sets, the verification set and the test set are different in the setting of the maximum sentence length. At the same time, their model has many areas that need to be improved.In their future work, they will not only improve our methods and systems but also continue to pay attention to related code-mixing fields progress.  

Concept used:Bert model and dravidian language. 

Merits:The conjecture was  feasible. 

Demerits: No data enhancement method was used for the problem of data imbalance. 

**2.7] A Survey:Feature Selection and Machine Learning Methods for Tamil Text Classification,International Journal of Recent Technology and Engineering** 

**(IJRTE) by N. Rajkumar, T. S. Subashini, K. Rajan, V. Ramalingam in 2020.** 

The objective and aim of this survey was to issue a sketch for the Machine Learning algorithm of Tamil Text classification. Since the past half decade, almost all regional languages have increased in the size of their computerized repository. Tamil language also has captivated all types of information to online digital repositories. The text data in digital format both in offline and online mode is very huge, so there is a requirement for a document classification system to classify the document according to its class name. Document classification is a method that examines the text data assumed in the document and classifies the categories. Text being in Tamil language requires the challenges of Natural Language processing (NLP). This paper gives a survey of Tamil Text Classification works completed on Tamil Language content. The automated document classification is at the crossroads of Machine Learning (ML), NLP and Information Retrieval (IR). This study mainly gives readers fruitfully acquired the necessary important information about the needed algorithm and its associated techniques. Thus, I believe that through this study it is useful to other professionals and researchers to propose new techniques in the province of Tamil text classification. This study shows that supervised learning algorithms the decision tree (DT), Artificial Neural Network (ANN), k-nearest neighbor (kNN) classifier, N-gram, Naive Bayes (NB) and Deep Learning performed optimally to the Text Classification task.  

Concepts used:Tamil Text Classification, NLP, Machine Learning 

Merits:ML algorithms such as KNN,NB classifier, SVMs and DTs were extremely useful for Tamil text classification . 

Demerits: Deep  learning  technology approach in Tamil Text Classification was not introduced in this paper. 

**2.8] SANAD: Single-Label Arabic News Articles Dataset for Automatic Text Categorization by Omar Einea,Ashraf Elnagar,Ridhwan Al-Debsi in 2019.** 

In this paper the main concept involved here was the text Classification which was one of the most popular Natural Language Processing (NLP) tasks. Text classification (aka categorization) is an active research topic in recent years. However, much less attention was directed towards this task in Arabic, due to the lack of rich representative resources for training an Arabic text classifier. Therefore, they introduced a large Single-labeled Arabic News Articles Dataset (SANAD) of textual data collected from three news portals. The dataset was a large one consisting of almost 200k articles distributed into seven categories that they offer to the research community on Arabic computational linguistics.They  anticipate that the rich dataset would make a great aid for a variety of NLP tasks on Modern Standard Arabic (MSA) textual data, especially for single label text classification purposes. They presented the data in the raw form. SANAD is composed of three main datasets scraped from three news portals, which are AlKhaleej, AlArabiya, and Akhbarona. SANAD was made public. 

Concept used: Arabic, NLP,News articles & Single-label text classification 

Merits: Bag of Words and LDA are used 

Demerits:Uncorrelated topics (Dirichlet topic distribution cannot capture correlations). 

**2.9]Improving text categorization by using a topic model An International Journal (** 

**ACIJ )by Wongkot Sriurai, in November 2011.** 

In this paper,they have applied the topic model approach to cluster the words into a set of topics. The concept of topic model, the words (or terms) are clustered into the same topics. Given D is a set of documents composed of a set of words (or terms) W, T is a set of latent topics that were  created based on a statistical inference on the term set W. In this paper, the topic model was applied based on the Latent Dirichlet Allocation (LDA) algorithm to produce a probabilistic topic model from a web page dataset. Their main goal was to compare between the feature processing techniques of BOW and the topic model. They also applied  and compared two feature selection techniques: Information Gain (IG) and Chi Squared (CHI). Three text categorization algorithms: Naive Bayes (NB), Support Vector Machines (SVM) and Decision tree, were used for evaluation. The experimental results showed that the topic-model approach for representing the documents yielded the best performance based on F1 measure equal to 79% under the SVM algorithm with the IG feature selection technique. 

Concept used:Text categorization , Bag of Words,Topic-Model 

Merits: Support vector machine shows good accuracy 

Demerits:Neighboring web pages were not included into the classification approach.So that further improvement needed. 

**2.10] Text classification using Naive Bayes classifier by Johnson Kolluri and Shaik Razia in 2020.** 

In this paper the main concept involved here was text classification and statistical methods.In all the previous applications where the data plays an important role such as universities, businesses, research institutions, technology-intensive companies, and government funding agencies, maintaining irregular data is a big challenge. Text classification using machine learning and deep learning models were used to organize documents or data in a predefined set of classes/groups. So once the data is trained using the deep learning algorithms, the trained model is able to identify, predict and detect the data for categorizing it in classes/groups/topics.  

Concept used: Statistical methods,Machine learning and text classification 

Merits:The proposed method in this paper is very useful in Web content management, Search engines email filtering, spam detection, intent detection, topic labeling, tagging, categorization of data and sentiment analysis, etc. 

Demerits: Deep learning technology was not introduced in this paper. 

**2.3] ISSUES IN EXISTING SYSTEM** 

Time Taken for Embedding BERT model is high. Accuracy was not achieved up to the level. The terms with lower term weights are eliminated from the keyword list.:Uncorrelated topics (Dirichlet topic distribution cannot capture correlations). No data enhancement method was used for the problem of data imbalance. No stop words removal approach was used.  Focuses on only Five entities.Various Mathematical formulas are used and the data training is not accurate. RL and DL Does not achieve much accuracy compared to other classifiers.Accuracy was not up to the level. Using Spark NLP gives us advantages when we want to use a BERT Large model and process large amounts of 

data.Keywords for each group from the clustering model as can be seen in the form of table. 

**2.4]SUMMARY OF LITERATURE SURVEY** 

In  the  above  Literature  Survey,  the  algorithms  used  are  TF-IDF(Term  Frequency  –Inverse Document  Frequency),BERT  Model  and  Named  Entity  Recognition.The  term tf–idf stands for term frequency–inverse document frequency, it is a mathematical statistic that is planned to reflect how significant a word is to a record in a collection or corpus. The tf–idf esteem builds proportionally to the number of times a word shows up in the document. It is offset by the quantity of documents in the corpus that contain the word, which helps to adjust for the fact that a few words show up more often when all is said in done. tf–idf is one of the most well-known term- weighting  plans  today.  An  overview  led  in  2015  demonstrated  that  83%  of  text-based recommender frameworks in advanced libraries use tf–idf. It would be difficult to understand tf– idf together. So, let's understand each separately - 

**Term Frequency (tf) -** It gives us the recurrence of the word in each report in the corpus. It is the proportion of the number of times the word shows up in a report contrasted with the all-out the number of words in that record. It increments as the quantity of events of that word inside the record increments. 

**Inverse Data Frequency (idf) -** It is used to figure the heaviness of uncommon words over all reports in the corpus. The words that happen seldom in the corpus have a high IDF score. 

BERT, which stands for Bidirectional Encoder Representations from Transformers, is based on Transformers, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection. The BERT algorithm is proven to perform 11 NLP tasks efficiently. It's trained on 2,500 million Wikipedia words and 800 million words of the Book Corpus dataset. Google Search is one of the most excellent examples of BERT's efficiency.BERT is so good Because BERT practices to predict missing words in the text, and because it analyzes every sentence with no specific direction, it does a better job at understanding the meaning of homonyms than previous NLP methodologies, such as embedding methods. ... So far, it's the best method in NLP to understand context-heavy texts. The BERT   model uses 12 layers of transformers block with a hidden size of 768 and number of self- attention heads as 12 and has around 110M trainable parameters. 

Named entity recognition (NER) , also known as entity chunking/extraction , is a popular 

technique used in information extraction to identify and segment the named entities and classify or categorize them under various predefined classes. In any text document, there are particular terms that represent specific entities that are more informative and have a unique context. These entities are known as named entities , which more specifically refer to terms that represent real- world objects like people, places, organizations, and so on, which are often denoted by proper names. A naive approach could be to find these by looking at the noun phrases in text documents. Named  entity  recognition  (NER) ,  also  known  as  entity  chunking/extraction ,  is  a  popular technique used in information extraction to identify and segment the named entities and classify or categorize them under various predefined classes. 

PAGE16 
**CHAPTER PAGE17** 

**3] SYSTEM DESIGN 3.1]INTRODUCTION** 

The proposed method involves three major stages namely data pre-processing, feature extraction and classification.Once the Tamil news articles are processed feature extraction process is carried out to derive an useful set of features with the help of Countvectorizer. The converted vectors are grouped using Bag-of-words.Finally, the multinomial naive bayes model is utilized as a classification model to identify the appropriate categories and the class labels of the particular dataset.So in this project why we are using  Naive Bayes is to categorize the Tamil news articles in an efficient way. With the help of this Naive Bayes technique,the dataset and the trained model we can achieve the desired output within the stipulated time. So that with the help of the trained model,we can achieve the desired accuracy at a higher level in which Inference is done for evaluating the model's performance and its accuracy. 

**3.2] SYSTEM ARCHITECTURE 3.2.1] FLOW CHART** 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.002.jpeg)

**Figure 1. System Architecture Flow Diagram** 

PAGE17 

**3.2.2] DESCRIPTION** 

The first step  after collecting the dataset using the net source,which is in raw format,we have to load that particular dataset.The command used to load the dataset is Command data() in which it will list all the datasets in loaded packages.For example,the command data(phones) will load the data set phones into memory.After loading the dataset by using the read\_csv() function from the pandas package, we can import tabular data from CSV files into pandas dataframe by specifying a parameter value for the file name (e.g. pd. read\_csv("filename. csv") ). 

The second step is to pre-process the data in which data preprocessing is that step in which the data gets transformed, or Encoded, to bring it to such a state that now the machine can easily parse it. In other words, the features of the data can now be easily interpreted by the algorithm.Data preprocessing is an integral step in Machine Learning as the quality of data and the useful information that can be derived from it directly affects the ability of our model to learn; therefore, it is extremely important that we preprocess our data before feeding it into our model.Data preprocessing is extremely important because it allows improving the quality of the raw experimental data.The first step in Data Preprocessing is to understand your data.Then Using  statistical methods or pre-built libraries that help us to visualize the dataset and give a clear image of how our data looks in terms of class distribution.Then Summarize your data in terms of the number of duplicates, missing values, and outliers present in the data.After which drop the fields which we think have no use for the modeling or are closely related to other attributes.Then dimensionality reduction is one of the very important aspects of data Preprocessing.Then at last we have to do some feature engineering to figure out which attributes contribute most towards model training. 

Then the third step is to split the dataset into as train and test data.The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model.It is a fast and easy procedure to perform, the results of which allow you to compare the performance of machine learning algorithms for your predictive modeling problem. Although simple to use and interpret, there are times when the procedure should not be used, such as when you have a small dataset and situations where additional configuration is required, such as when it is used for classification and the dataset is not 

PAGE18 

balanced.The train-test split procedure is appropriate when you have a very large dataset, a costly model to train, or require a good estimate of model performance quickly. 

Train dataset is used to fit the machine learning model whereas the test dataset is used to evaluate the fit machine learning model. 

`                          `The fourth step is feature extraction.A feature is a distinctive attribute or aspect of something (so this can be something abstract or apprehensible, conceptual or physical). This term is commonly used in machine learning, pattern recognition and image processing, where to describe how from a measured set of data, values (features) are derived seeking to be informative and non-redundant.In linguistics, a feature is a distinctive characteristic of a linguistic unit (especially a speech sound or vocabulary item) that serves to distinguish it from others of the same type.So, feature extraction is that mechanism, algorithm, process, formula or procedure that let us build, derive or identify (and collect) that distinctive characteristic or aspects from a text.Feature extraction step means to extract and produce feature representations that are appropriate for the type of NLP task that we are trying to accomplish and the type of model which we are planning to use.Feature extraction helps us to reduce the amount of redundant data from the data set. In the end, the reduction of the data helps to build the model with less machine's efforts and also increases the speed of learning and generalization steps in the machine learning process.Briefly, NLP is the ability of computers to understand human language. Machine Learning algorithms learn from a predefined set of features from the training data to produce output for the test data. ... So, we need some feature extraction techniques to convert text into a matrix(or vector) of features. 

`                                 `The fifth step is once the feature extraction is performed then training the model will be performed which is nothing but the multinomial naive bayes classifier will be trained.Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.Naive Bayes classifier is a collection of many algorithms where all the algorithms share one common principle, and that is each feature being classified is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of the other feature. 

The Naive Bayes algorithm has the following advantages:It is easy to implement as you only have 

to calculate probability.You can use this algorithm on both continuous and discrete data.It is simple and can be used for predicting real-time applications.It is highly scalable and can easily handle large datasets.So that’s why we have chosen the naive bayes algorithm for our implementation. 

`                            `The sixth step is to evaluate the models performance here the model will be tested and the required results will be analyzed in the form of performance metrics parameters such as F1 score,precision,recall and accuracy..Precision and recall were used to assess the system’s classification accuracy, i.e. the fraction of correctly categorised news documents. The model’s accuracy tells us how good it is at spotting things. There are two types of people in our world:those who are positive and those who are negative. Precision informs us about the likelihood of achieving a right positive result. The categorising of classes. The model’s recall indicates how sensitive it is to detecting the positive class. 

`                                     `At last inference is performed which is nothing but Inference will be performed once the naive bayes classifier has been trained. Inference is the process of making a prediction using a trained machine learning system. Data can be fed into a trained machine learning model, allowing predictions to be made that can be used to guide decision logic on the device or at the edge gateway.Inference basically is using observation and background to reach a logical conclusion. You probably practice inference every day. For example, if you see someone eating a new food and he or she makes a face, then you infer he does not like it. Or if someone slams a door, you can infer that she is upset about something.To be precise inference is the stage in which a trained model is used to infer/predict the testing samples and comprises a similar forward pass as training to predict the values. 

**3.3]SYSTEM REQUIREMENTS** 

Requirements are defined during the early stages of the system development as a specification of what should be implemented. A collection of requirements is a requirements document. They may be user level facility description, detailed specification of system behavior , general system property, a specific constraint on the system or information on how to carry on computation. 

`                                         `System requirements is a statement that identifies the functionality that is needed by a system in order to satisfy the customer's requirements.Basically there are three types of requirements : system requirements,functional requirements and non-functional requirements. 

The system requirements needed for our research are as follows: 1)A raw dataset: 

`          `Raw data is the data that is collected from a source, but in its initial state. It has not yet been processed — or cleaned, organized, and visually presented. Raw data can be manually written down or typed, recorded, or automatically input by a machine. You can find raw data in a variety of places, including databases, files, spreadsheets, and even on source devices, such as a camera. Raw data is just one type of data with potential energy. 

Here are some examples of data in raw form: 

-->A list of every purchase at a store during a month but with no further structure or analysis. -->Every second of footage recorded by a security camera overnight. 

-->The grades of all of the students in a school district for a quarter. 

-->A list of every movie being streamed by a video streaming company. 

-->Open-ended responses to a survey question. 

2)Python 3: 

Python is a high-level, interpreted, interactive and object-oriented scripting language. Python is designed to be highly readable. It uses English keywords frequently whereas the other languages use punctuations. It has fewer syntactic constructions than other languages. 

- Python is Interpreted − Python is processed at runtime by the interpreter. You do not need to compile your program before executing it. This is similar to PERL and PHP.
- Python is Interactive − You can actually sit at a Python prompt and interact with the interpreter directly to write your programs.
- Python is Object-Oriented − Python supports Object-Oriented style or technique of programming that encapsulates code within objects.
- Python is a Beginner's Language − Python is a great language for the beginner-level programmers and supports the development of a wide range of applications from simple text processing to WWW browsers to games.

Python has a big list of good features.A few are listed below,  

- It supports functional and structured programming methods as well as OOP.
- It can be used as a scripting language or can be compiled to byte-code for building large applications.
- It provides very high-level dynamic data types and supports dynamic type checking.
- It supports automatic garbage collection.
- It can be easily integrated with C and C++

3)Google Colab: 

`    `Google Colab is a web IDE used for python..Colaboratory, or “Colab” for short, is a product from Google Research. Colab allows anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education.Colab is a free notebook environment that runs entirely in the cloud. It lets you and your team members edit documents, the way you work with Google Docs. Colab supports many popular machine learning libraries which can be easily loaded in your notebook.Colab supports GPU and it is totally free. The reasons for making it free for the public could be to make its software a standard in the academics for teaching machine learning and data science. It may also have a long term perspective of building a customer base for Google Cloud APIs which are sold on a per-use basis. 

`                      `Google Colab is a powerful platform for learning and quickly developing machine learning  models  in  Python.  It  is  based  on  the  Jupyter  notebook  and  supports  collaborative development. The team members can share and concurrently edit the notebooks, even remotely. The notebooks can also be published on GitHub and shared with the general public. Colab supports many popular ML libraries such as PyTorch, TensorFlow, Keras and OpenCV. The restriction as of today is that it does not support R or Scala yet. There is also a limitation to sessions and size. Considering the benefits, these are small sacrifices one needs to make. 

**3.4]SUMMARY** 

`    `Here we have seen all about how the system architecture was formed and then how the flow of system architecture have been put up and then the working of system architecture such as loading the dataset,preprocessing the dataset,splitting the data sets into train and test data also we have seen the most important flow of system architecture called the feature extraction.Then we have seen about the model naive bayes classifier which has been induced in our project.At last we have also discussed about the system requirements such as the raw dataset,about the language used for our project which is called as python 3 and also have discussed about the web IDE used to code for the python which is nothing but the google colab.So that’s all about the summary of our  system design.

`                                                        `**CHAPTER 4 4.]MODULE DESCRIPTION 4.1]INTRODUCTION** 

`   `Considering the proposed system and our objective in which it is to Categorize the Tamil news articles available by Naive Bayes model using Natural Language Processing.So basically our module mainly consists of four stages such as the first module is scraping the data and then in the second module text pre-processing is performed after which  in the third module training and testing of the dataset and also hyperparameter tuning is done and atlast coming to the fourth module with the help of bag of words and trained model inference is performed. 

**4.2]SCRAPPING THE DATA:** 
**
` `Data scraping, also known as web scraping, is the process of importing information from a website into a spreadsheet or local file saved on your computer. It's one of the most efficient ways to get data from the web, and in some cases to channel that data to another website.To be precise, scraping is the act of extracting data or information from websites with or without the consent of the website owner. Scraping can be done manually, but in most cases it's done automatically because of its efficiency. 

A web scraping tool will load the URLs given by the users and render the entire website. As a result, you can extract any web data with simple point-and-click and file in a feasible format into your computer without coding. For example, you might want to extract posts and comments from Twitter.So this is how scraping works and it is being implemented in our day to day lives.Here for our project,especially for data scraping we have used a data extraction software called an Octoparse. 

`                                                          `Octoparse is a modern web data extraction software with a visual interface. Octoparse is simple to use for both expert and beginner users to bulk extract information from websites; for most scraping activities, no code is required. Octoparse enables getting data from the web easier and faster without requiring you to code. It will automatically extract content from practically any page and save it in a format of your choice as clean structured data. You can also create bespoke APIs from any data. You no longer need to hire a slew of interns to manually copy and paste. You only need to create a data collection rule, and Octoparse will take care of the rest. Octoparse 8 is used to scrape news items.Web scraping, also known as web harvesting or web data extraction, is a type of data scraping that is used to gather information from websites. 

`                      `Using the Hypertext Transfer Protocol or a web browser, web scraping software can directly access the World Wide Web. While a software user can perform web scraping manually, the word usually refers to automated procedures carried out by a bot or web crawler. 

It's a type of copying in which specific data is acquired and copied from the internet, usually into a central local database or spreadsheet for retrieval or analysis later. 

Web scraping is the process of retrieving a web page and extracting information from it. 

Fetching is the process of downloading a webpage (which a browser does when a user views a page).As a result, web crawling is an important part of web scraping, as it allows you to collect pages for subsequent processing. 

After the data has been fetched, extraction can begin. 

A page's content can be analysed, searched, reformatted, and the data put into a spreadsheet or a database. 

Web scrapers often extract information from a page in order to use it for another purpose. Finding and copying names and phone numbers, companies and their URLs, or e-mail addresses to a list is an example (contact scraping). 

`                             `Web scraping is used for contact scraping and as part of applications for web indexing, web mining and data mining, online price change monitoring and price comparison, product review scraping (to keep an eye on the competition), real estate listing gathering, weather data monitoring, website change detection, research, tracking online presence and reputation, web mashup, and web data integration.Text-based mark-up languages (HTML and XHTML) are used to create web pages, and they usually contain a plethora of important data in the form of text. Most online sites, on the other hand, are created for human end-users, not for automated usage. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.003.jpeg)

**Figure 2. Data** 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.004.png)

**Figure 3.Scrapping the data**

**4.3]TEXT PRE-PROCESSING** 

In Natural language processing,text preprocessing is the first step in the process of building a model.Text preprocessing is a method to clean the text data and make it ready to feed data to the model.Text preprocessing steps are widely used for dimensionality reduction.Text preprocessing is traditionally an important step for natural language processing (NLP) tasks. It transforms text into a  more  digestible  form  so  that  machine  learning  algorithms  can  perform  good  and better.Preprocessing the text is very much important because it helps to get rid of unhelpful parts of the data,or noise, by converting all characters to lowercase, removing punctuations marks, and removing stop words and typos. Removing noise comes in handy when you want to do text analysis on pieces of data like comments or tweets.The main objective of text pre-processing techniques are to transform the unstructured or the semi structured data or it can be even a text data into a structured data model.There are many natural language processing techniques used for pre-processing the textual  data.Here,for  our  project  implementation  we  have  used  few  NLP  techniques  such  as tokenization,lemmatization,stemming,stopwords removal,vectorization and Label Encoding. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.005.jpeg)

`                                        `**Figure 4.Text pre-processing 4.3.1]TOKENIZATION** 

Tokenization is a step which splits longer strings of text into smaller pieces, or tokens. Larger chunks of text can be tokenized into sentences, sentences can be tokenized into words, etc. Further processing  is  generally  performed  after  a  piece  of  text  has  been  appropriately  tokenized. Tokenization is also referred to as text segmentation or lexical analysis. Sometimes segmentation is used to refer to the breakdown of a large chunk of text into pieces larger than words (e.g. paragraphs or sentences), while tokenization is reserved for the breakdown process which results exclusively in words.Tokenization is the first step in any NLP pipeline. It has an important effect on the rest of your pipeline. A tokenizer breaks unstructured data and natural language text into chunks of information that can be considered as discrete elements. The token occurrences in a document can be used directly as a vector representing that document.This immediately turns an unstructured string (text document) into a numerical data structure suitable for machine learning. They can also be used directly by a computer to trigger useful actions and responses. Or they might be  used  in  a  machine  learning  pipeline  as  features  that  trigger  more  complex  decisions  or behavior.Tokenization is commonly used to protect sensitive information and prevent credit card fraud. ... The real bank account number is held safe in a secure token vault.Tokenization can be done to either separate words or sentences. If the text is split into words using some separation technique it is called word tokenization and the same separation done for sentences is called sentence tokenization.There are various tokenization techniques available which can be applicable based on the language and purpose of modelingOne of the biggest challenges in the tokenization is the getting the boundary of the words. In English the boundary of the word is usually defined by a space and punctuation marks define the boundary of the sentences, but it is not the same in all the languages. In languages such as Chinese, Korean, and Japanese symbols represent the words and it is difficult to get the boundary of the words. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.006.jpeg)

`                                      `**Figure 5.Tokenization** .**4.3.2]LEMMATIZATION** 

Lemmatization  usually  refers  to  doing  things  properly  with  the  use  of  a  vocabulary  and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.In other words, Lemmatization is a method responsible for grouping different inflected forms of words into the root form, having the same meaning. It is similar to stemming, in turn, it gives the stripped word that has some dictionary  meaning.Lemmatization generally  means to  do the things properly with the use of vocabulary and morphological analysis of words.Lemmatization is the process of converting a word to its base form. The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language. ![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.007.png)In Lemmatization the root word is called Lemma. ... For example, runs, running, ran are all forms of the word run, therefore run is the lemma of all these words.A trivial way to do lemmatization is by simple dictionary lookup. This works well for straightforward inflected forms, but a rule-based system will be needed for other cases, such as in languages with long compound words.The general rule  for  whether  to  lemmatize  is  unsurprising:  if  it  does  not  improve  performance,  do  not lemmatize.Lemmatization is slower as compared to stemming but it knows the context of the word before  proceeding.It  is  a  dictionary-based  approach.Accuracy  is  more  as  compared  to Stemming.Lemmatization would be recommended when the meaning of the word is important for analysis. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.008.jpeg)

`                                            `**Figure 6.Lemmatization 4.3.3]STEMMING** 

Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma. Stemming is important in natural language understanding (NLU) and natural language processing (NLP).Stemming is basically removing the suffix from a word and reducing it to its root word. For example: “Flying” is a word and its suffix is “ing”, if we remove “ing” from “Flying” then we will get the base word or root word which is “Fly”.Stemming is a technique used to extract the base form of the words by removing affixes 

from them. It is just like cutting down the branches of a tree to its stems. 

`             `For example, the stem of the words eating, eats, eaten is eat. Search engines use stemming for indexing the words.A stemming algorithm is a process ![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.009.png)of linguistic normalisation, in which the variant forms of a word are reduced to a common form, for example, 

connection 

connections 

connective          --->   connect 

connected 

connecting 

It is important to appreciate that we use stemming with the intention of improving the performance of IR systems. It is not an exercise in etymology or grammar. In fact from an etymological or grammatical viewpoint, a stemming algorithm is liable to make many mistakes. In addition, stemming algorithms - at least the ones presented here - are applicable to the written, not the spoken, form of the language.Stemming is faster because it chops words without knowing the context of the word in given sentences.It is a rule-based approach.Accuracy is less.Stemming is preferred when the meaning of the word is not important for analysis.Example:spam detection.Stemming is used in information retrieval systems like search engines.It is used to determine domain vocabularies in domain analysis. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.010.jpeg)

`                                     `**Figure 7.Stemming 4.3.4]STOPWORDS REMOVAL** 

Stop words are a set of commonly used words in a language. Examples of stop words in English are “a”, “the”, “is”, “are” and etc. Stop words are commonly used in Text Mining and Natural Language Processing (NLP) to eliminate words that are so commonly used that they carry very 

little useful information.The general strategy for determining a stop list is to sort the terms by collection frequency (the total number of times each term appears in the document collection), and then to take the most frequent terms, often hand-filtered for their semantic content relative to the domain of the documents being indexed, as a stop list,the members of which are then discarded during indexing.To remove stop words from a sentence, you can divide your text into words and then remove the word if it exists in the list of stop words provided by NLTK. In the script above, we first import the stopwords collection from the nltk. corpus module. Next, we import the word\_tokenize() method from the nltk.We should use stop words For tasks like text classification, where the text is to be classified into different categories, stopwords are removed or excluded from the given text so that more focus can be given to those words which define the meaning of the text.Stop words are often removed from the text before training deep learning and machine learning models since stop words occur in abundance, hence providing little to no unique information that can be used for classification or clustering.On removing stopwords, dataset size decreases, and the time to train the model also decreases without a huge impact on the accuracy of the model. Stopword removal can potentially help in improving performance, as there are fewer and only significant tokens left. Thus, the classification accuracy could be improved.Stop words are just a set of commonly used words in any language. Stop words are commonly eliminated from many text processing applications because these words can be distracting, non-informative and are additional memory overhead. 

Stop words are available in abundance in any human language. By removing these words, we remove the low-level information from our text in order to give more focus to the important information. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.011.png)

`                                                  `**Figure 8.Stopwords removal 4.3.5]VECTORIZATION** 

Vectorization is basically the art of getting rid of explicit for loops in your code. In the deep learning era, with safety deep learning in practice, you often find yourself training on relatively large data sets, because that's when deep learning algorithms tend to shine.Vectorization is the process of converting an algorithm from operating on a single value at a time to operating on a set of values (vector) at one time. Modern CPUs provide direct support for vector operations where a single instruction is applied to multiple data (SIMD).Basically Text Vectorization is the process of converting text into numerical representation. Here are some popular methods to accomplish text vectorization: Binary Term Frequency. Bag of Words (BoW) Term Frequency. (L1) Normalized Term Frequency.Vectorization is important in Machine Learning because Just like in the real- world we are interested in solving any kind of problem efficiently in such a way that the amount of error is reduced as much as possible.In machine learning, there’s a concept of an optimization algorithm that tries to reduce the error and computes to get the best parameters for the machine learning model.So by using a vectorized implementation in an optimization algorithm we can make the process of computation much faster compared to Non Vectorized Implementation. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.012.png)

`                                           `**Figure 9.Vectorization 4.3.6]LABEL ENCODING** 

Label Encoding refers to converting the labels into a numeric form so as to convert them into the machine-readable form. Machine learning algorithms can then decide in a better way how those labels must be operated. It is an important preprocessing step for the structured dataset in supervised learning.Label encoding is applied when the categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school) and when the number of categories is quite large as one-hot encoding can lead to high memory consumption.Label encoding converts the data in machine- readable form, but it assigns a unique number(starting from 0) to each class of data. This may lead to the generation of priority issues in the training of data sets. A label with a high value may be considered to have higher priority than a label having a lower value.The advantages of label encoding are such as it is easy to implement and interpret and it is visually user friendly and also it works best with a smaller number of unique categorical values.To be precise label encoding is a popular encoding technique for handling categorical variables.In this technique, each label is assigned a unique integer based on alphabetical ordering. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.013.jpeg)

**Figure 10.Label encoding** 

**4.4]TRAINING AND TESTING DATA AND HYPERPARAMETER TUNING 4.4.1]FLOWCHART ![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.014.png)**

Scrapped Data![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.015.png)![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.016.png)

Splitting of Data Into Train,Test and Cross Validation![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.015.png)

Tuning the Dataset by Hyperparameter Tuning using GridsearchCV![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.015.png)

**Figure 11.Flow chart of training and testing data** 

**4.4.2]DESCRIPTION**  

Data is used to train machine learning algorithms.They use the training data to form associations, gain insight, make judgments, and assess their confidence.The model works better when the training data is good. 

In reality, the quality and amount of your machine learning training data is just as important as the algorithms themselves in determining the success of your data project. 

First and foremost, we must all agree on what we mean when we say "dataset." 

A dataset is defined as a collection of rows and columns, with each row having one observation. A picture, an audio clip, text, or video can be used to represent this observation. 

There are several aspects to consider when determining how much machine learning training data you require. 

The importance of accuracy is the first and primary consideration. 

Let's pretend you're working on a sentiment analysis algorithm. 

A sentiment algorithm with an accuracy of 85 to 90% is more than adequate for most people's needs, and a few false positives or negatives here and there won't make much of a difference. Which would you rather have: a cancer detection model or a self-driving vehicle algorithm? That's another thing altogether. 

It's truly a matter of life and death if a cancer detection model misses critical indications. The model is first fitted using a training data set, which is a collection of instances used to learn the model's parameters (for example, the weights of connections between neurons in artificial neural networks).A huge dataset used to educate a machine learning model is referred to as training data.The training data for supervised machine learning models is labelled. Unsupervised machine learning models are trained on data that is not labelled. 

The concept of using training data in machine learning systems is a basic one, yet it is fundamental to how these technologies operate.The training data is a set of data used to teach a software how to learn and deliver advanced results using technologies such as neural networks. It can be supplemented with additional data sets known as validation and testing sets. 

A training set, a training dataset, or a learning set are all terms used to describe training data. 

The training set is the material that the computer uses to learn how to analyse data. Machine learning employs algorithms to simulate the human brain's ability to take in a variety of inputs and weigh them in order to generate activations in the brain's individual neurons. Software – machine learning and neural network programmes that give incredibly precise simulations of how our human cognitive processes function – replicates a lot of this process 

Data that has been explicitly identified for use in testing, usually of a computer programme, is referred to as test data.Some data can be utilised in a confirmatory manner, for example, to ensure that a particular set of inputs to a function provides the desired output.Other information might be utilised to test the program's capacity to respond to uncommon, severe, exceptional, or unexpected input.Test data can be generated in a systematic or targeted manner (as is generally the case in domain testing), or in a more ad hoc manner (as is typically the case in high-volume randomised automated tests).The tester or a software or function that assists the tester can generate test data. 

Cross-validation, also known as rotation estimation or out-of-sample testing, is a 

collection of model validation procedures for determining how well the findings of a statistical investigation will generalise to a different set of data.Cross-validation is a resampling approach that tests and trains a model using various chunks of the data on successive rounds.It's most commonly employed in situations when the aim is prediction and the user wants to know how well a predictive model will perform in practise. 

GridSearchCV  is the process of fine-tuning hyperparameters to find the best values for a certain model. As previously stated, the value of hyperparameters has a substantial impact on a model’s performance. It’s worth noting that there’s no way to know ahead of time what the best values for hyperparameters are, therefore we should try all of them to find the best ones. Because manually adjusting hyperparameters would take a significant amount of time and resources, we use GridSearchCV to automate the process. Grid Search calculates the performance for each combination of all the supplied hyperparameters and their values, and then chooses the optimum value for the hyperparameters. Based on the amount of hyperparameters involved, this makes the processing time-consuming and costly. GridSearchCV does cross-validation in addition to grid search.  

`                    `The model is trained using cross-validation. As we all know, we divide the data into two pieces before training the model with it: train data and test data. The procedure of cross- validation divides the train data into two parts: the train data and the validation data.K-fold Cross- validation is the most common type of cross-validation. The train data is divided into k divisions using an iterative approach. One division is kept for testing and the remaining k-1 partitions are used to train the model in each iteration. In the next iteration, the next partition will be used as test data, and the remaining k-1 will be used as train data, and so on. It will record the model’s performance in each iteration and offer the average of all the results in the end. As a result, it is a time-consuming operation. As a result, evaluating the optimum hyperparameters using GridSearch and cross-validation takes a lengthy time. Comparing the results of Tuned and Untuned Models is always a good idea. This will take time and money, but it will undoubtedly yield the best results. If you need assistance, the scikit-learn API is a fantastic place to start. Learning by doing is always beneficial.

**4.5]BAG OF WORDS,TRAINING THE MODEL AND INFERENCE 4.5.1]FLOWCHART ![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.017.png)**

Splitted Dataset Bag ![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.018.png)Of Words![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.019.png)

Naive Bayes Classifier Inference![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.019.png)![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.020.png)

`               `**Figure 12.Flow chart for training the model 4.5.2]DESCRIPTION** 

The bag-of-words model is a representation for natural    language processing and information retrieval that simplifies things . A text (such as a sentence or a document) is represented 

in this paradigm as a bag (multiset) of its words, which ignores syntax and even word order while maintaining multiplicity. Computer vision has also employed the bag-ofwords 

concept.The bag-of-words model is widely used in document classification approaches, where the (frequency of) occurrence of each word is utilised to train a classifier 

The Naive Bayes classifier is based on the Bayes theorem’s conditional probability principle .The Naive Bayes classifier is based on the Bayes theorem’s conditional probability principle . We use Bayesian likelihood to reason backwards to find the events or random causes that most likely caused a specific result. These arbitrary factors in this content arrangement model will be the terms in the archive and their frequencies. Word recurrence is the component on which a multinomial guileless Bayes classifier bases its calculation .The Bayes’ Theorem-based Naive Bayes classifier is a classification algorithm based on Bayes’ Theorem. It is a family of algorithms rather than a single method, and they all follow the same principle: each pair of features to be classified is independent of the others. The Nave Bayes classifier is one of the most basic ways to classification that can nevertheless provide good accuracy. It’s a probabilistic classifier that’s built on probabilistic models with strong independence assumptions. The model will distribute the report to the class, classification, as a result of the outcome.You use Bayesian. System Architecture of the proposed model 

likelihood to reason backwards to find the events or random causes that most likely caused a specific result. These arbitrary factors in this content arrangement model will be the terms 

in the archive and their frequencies. Word recurrence is the component on which a multinomial guileless Bayes classifier bases its calculation . The model will distribute the report 

to the class, classification, as a result of the outcome. Naïve Bayes can be calculated by 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.021.png)

**Figure 13.Naïve Bayes Formula** 

Where, 

A is the  probability of  x\_train\_bow  B is  the probability of  y\_train\_label 

A supervised learning approach, such as gradient descent or stochastic gradient descent, is used to train the model (for example, a naïve Bayes classifier) using the training data set. 

In practise, the training data set is often made up of pairs of input vectors (or scalars) and output vectors (or scalars), with the response key referred to as the goal (or label). 

Model evaluation is an important step in the creation of a model. 

It aids in the selection of the best model to represent our data and the prediction of how well the chosen model will perform in the future. 

In data science, evaluating model performance with the data used for training is not a good idea since it might lead to overoptimistic and overfitted models. 

Hold-Out and Cross-Validation are two strategies for testing models in data science. 

Both techniques employ a test set (not visible to the model) to evaluate model performance in order to avoid overfitting.When just a little quantity of data is available, we employ k-fold cross- validation to provide an unbiased assessment of the model's performance. 

The number of classifications a model successfully predicts divided by the total number of predictions is known as model accuracy. 

It's one method of evaluating a model's performance, but it's far from the only one. 

In fact, there are many rich measurements that can be used for this purpose, and when you consider many of them at once rather than just one, accuracy gives you the best picture of how well a model is performing on a given dataset. There are many cases where a first pass at model training is insufficient, and the accuracy needs to be increased. 

This is a common occurrence in machine learning. 

**4.6]SUMMARY** 

As a result, evaluating the optimum hyperparameters using GridSearch and cross- validation takes a lengthy time. Comparing the results of Tuned and Untuned Models is always a good idea. This will take time and money, but it will undoubtedly yield the best results. If you need assistance, the scikit-learn API is a fantastic place to start. Learning by doing is always beneficial. The Multinomial Naive Bayes algorithm is a probabilistic learning approach popular in Natural Language Processing (NLP).  

The programme guesses the tag of a text, such as an email or a newspaper storey, using the Bayes theorem. It calculates each tag’s likelihood for a given sample and outputs the tag with the highest probability. The Naive Bayes method is a strong tool for analysing text input and solving problems with numerous classes. Because the Naive Bayes theorem is based on the Bayes theorem, it is necessary to first comprehend the Bayes theorem notion. Inference is completed when the na¨ıve bayes classifier has been trained. 

` `Inference is the process of making a prediction using a trained machine learning system. Data can be fed into a trained machine learning model, allowing predictions to be made that can be used to guide decision logic on the device or at the edge gateway.                                As previously stated, the value of hyperparameters has a substantial impact on a model’s performance. It’s worth noting that there’s no way to know ahead of time what the best values for hyperparameters are, therefore we should try all of them to find the best ones. Because manually adjusting hyperparameters would take a significant amount of time and resources, we use GridSearchCV to automate the process.  

`                                                       `**CHAPTER 5 5]SYSTEM** 

**IMPLEMENTATION** 

**5.1] INTRODUCTION** 

Businesses are being pushed to incorporate new methodologies and resources to facilitate improved navigation, processing, and management of high-dimensional data by a tremendous increase in web-based online content.90 percent of the data on the Internet is unstructured, and there are various methods for converting this data into valuable, organised data—classification is one of them.Knowledge classification into a useful set of groupings is important and vital. Automatic text classification is becoming increasingly important as the amount of machine- readable texts grows.The purpose of news classification is to assign categories to news documents based on their content. In order to accomplish this, we take the following steps:  

1) A selection of internet news sites is made depending on their nature and popularity. 

   2) We crawl each site with Octoparse 8 and build a Dataset with title, URL, description, and date. 
1) After receiving each unique document, we use a variety of approaches to process the text, including tokenization, punctuation removal, digit removal, and so on.  
1) We next do some complex text processing tasks, such as removing single letter words and stopping words.  
1) To improve accuracy, the dataset is divided into three sections: train, set, and cross-validation. 
1) Bag of Words is a tool that allows you to collect similar word embeddings and save them as common embeddings. 

   2) The Label Encoder is a machine-readable code converter. 
   2) The naive bayes classifier model is optimised using GridSearch.  
1) Finally, each document is treated as a word vector, with supervised learning used to assign many relevant classifications.  
1) The results of the supervised learning task are applied to new articles, and we use the Naive- Bayes classifier to categorise the papers. 

**5.2]OVERVIEW OF THE PLATFORM 5.2.1] GOOGLE COLAB** 

In terms of AI research, Google is fairly active. 

Google spent years developing TensorFlow, an AI framework, and Colaboratory, a development platform.TensorFlow is now open-source, and Google has made Colaboratory free to use since 2017. 

Google Colab, or simply Colab, is the new name for Colaboratory. 

The utilisation of GPU is another appealing feature that Google provides to developers. 

Colab is a free application that supports GPU.Its software might become a standard in academia for teaching machine learning and data science if it is made freely available to the public.It might also have the long-term goal of establishing a client base for Google Cloud APIs, which are sold on a per-use basis.There are various advantages of using Google Colab rather to a standard Jupyter Notebook instance.Collaboration with Pre-Installed Libraries in the Cloud,use of the GPU and TPU for free. 

The  Anaconda  package  of  Jupyter  Notebook  came  pre-installed  with  various  data libraries, including Pandas, NumPy, and Matplotlib, which is fantastic. 

Google Colab, on the other hand, comes with even more machine learning libraries pre-installed, including Keras, TensorFlow, and PyTorch. 

Everything is stored on your local PC when you use a standard Jupyter notebook as your development environment. 

This may be a desirable feature for you if you are concerned about your privacy. 

Google Colab, on the other hand, is the way to go if you want your notebooks to be available from any device with a simple Google log-in. 

All of your Google Colab notebooks, like your Google Docs and Google Sheets files, are kept in your Google Drive account. 

The collaboration function of Google Colab is another fantastic feature. 

If you're working on a project with numerous developers, Google Colab notebook is a terrific tool to utilise.You can co-code with many developers using a Google Colab notebook, just like you can with a Google Docs page.In addition, you may share your finished work with other developers. 

I believe that using Google Colab instead of a local Jupyter notebook is a no-brainer. You may use Google Research's dedicated GPUs and TPUs for your own machine learning projects. From personal experience, GPU and TPU acceleration make a significant impact in some applications, even tiny ones. 

This is one of the key reasons I use Google Colab to code all of my instructional projects. Furthermore, because it uses Google resources, the neural network optimization procedures have no effect on my CPUs, and my cooling fan does not spin up. 

**5.2.2] PYTHON** 

Python is one of the most widely 

used programming languages in the 

world.

The majority of top U.S. institutions employ Python in their beginning coding programmes, teaching students how to use it to create simple games, analyse data from web pages, and even do language processing. Python is a high-level, general-purpose coding language, which means it's simple to learn and can be used to tackle a wide range of issues. 

Python is a beneficial language to learn and use for a number of applications due to its simple syntax, high readability, and applicability across operating systems. 

When it comes to data science and data analysis, Python is also a popular language.Python is a high-level, general-purpose programming language that is interpreted. 

The use of considerable indentation in its design philosophy promotes code readability. Its language elements and object-oriented approach are aimed at assisting programmers in writing clear, logical code for both small and large-scale projects. 

Processing data and inferring trends is what data science is all about, and Python is tremendously useful in this field thanks to libraries like scipy, numpy, and pandas. 

The matplotlib software may also be used to create data visualisations. 

Python is a popular choice in industries such as bioinformatics, which require a lot of data and modelling. 

Because of their simplicity of use and scalability, many Python-based solutions have grown in popularity. 

Netflix, for example, employs numerical computation using scipy and numpy to manage user traffic throughout the platform. 

Machine learning algorithms can detect patterns in large volumes of data and use those patterns to predict future behaviour. 

Recommendation systems are the most apparent use of machine learning in goods. 

Netflix, Spotify, and Youtube use user data in order to make predictions and recommend playlists and content. 

Scikit-learn and tensorflow are Python packages that allow you to apply classification, clustering, and regression methods on huge data sets. 

These libraries are vital in any machine learning investigation because to their great speed and comprehensive functionality. 

Artificial intelligence encompasses many different aspects, including machine learning. Artificial intelligence (AI) refers to the development of computer systems that can perform human-like activities including perception and decision-making. 

Ridesharing applications like Uber and Lyft are a good illustration of artificial intelligence in action. 

Uber uses artificial intelligence (AI) to anticipate consumer demand and predicted arrival times (among other things), and much of this is done with Python. 

Other Python libraries, like as keras and pytorch, can be used to develop AI capabilities such as prediction models and neural networks, in addition to some of the previously listed Python tools. 

Python is one of the most widely used programming languages, with a wide range of applications. Furthermore, it is profoundly ingrained in our daily lives as well as in today's major corporations. 

Python is used in a variety of jobs, including software engineering, web development, data science, product management, business analysis, and more. 

Companies in today's technology era, regardless of sector, rely on data-driven decision making, and Python is the ideal tool for doing so. 

Python is a wonderful choice if you or your kid are interested in any of the fields or occupations listed above. 

Python's simplicity is the first of several advantages in data research. 

While some data scientists have backgrounds in computer science or know other programming languages, many come from backgrounds in statistics, mathematics, or other technical subjects and may not have as much coding knowledge when they enter the profession. 

Python syntax is straightforward to understand and write, making it a quick and easy programming language to pick up. 

Furthermore, there are several free resources accessible online to help you learn Python and obtain assistance if you get stuck. Python is an open source language, which means it is freely available to the general public. 

**5.3]IMPLEMENTATION DETAILS 5.3.1]SIMULATION PARAMETERS** 

Octoparse is a modern web data extraction software with a visual interface. Octoparse is simple to use for both expert and beginner users to bulk extract information from websites; for most scraping activities, no code is required. Octoparse enables getting data from the web easier and faster without requiring you to code. It will automatically extract content from practically any page and save it in a format of your choice as clean structured data. 

You can also create bespoke APIs from any data. You no longer need to hire a slew of interns to manually copy and paste. You only need to create a data collection rule, and Octoparse will take care of the rest. Octoparse 8 is used to scrape news items. The following is a flow chart for extracting data from a news website in fig 2 The features in this study were chosen using the TF- IDF method. The Term Frequency-Inverse Document Frequency (TF-IDF)  algorithm is a widely used method for converting text into a comprehensible numerical representation. Stopwords filtering can be done with TF-IDF in a variety of applications, including text summarization and categorization.We must first handle text before we can utilise it to demonstrate. This entails removing stop words, lemmatizing, stemming, tokenization, and vectorization, among other things. Vectorization is the process of converting text data into a machine readable structure.  

Words are referred to as vectors. TFIDFVectorizer tokenizes(tokenization means breaking down a sentence, portion, or message into words) the message while also performing really basic preprocessing such as removing accent marks, turning all of the words to lowercase, and so on. The jargon of learned words is formed, which will later be used to encode a hidden message. A vector that has been encoded.The bag-of-words model is a representation for natural language processing and information retrieval that simplifies things . A text (such as a sentence or a document) is represented in this paradigm as a bag (multiset) of its words, which ignores syntax and even word order while maintaining multiplicity. Computer vision has also employed the bag- of words concept.The bag-of-words model is widely used in document classification approaches, where the (frequency of) occurrence of each word is utilised to train a classifier. The Sklearn Library can be used to do label encoding in Python. Sklearn is a powerful tool for converting categorical feature levels into numerical values. LabelEncoder uses a value between 0 and n classes-1 to encode labels, where n is the number of different labels. When a label is repeated, it is given the same value as before. 

` `Grid SearchCV is the process of fine-tuning hyperparameters to find the best values for a certain model.Grid Search calculates the performance for each combination of all the supplied hyperparameters and their values, and then chooses the optimum value for the hyperparameters. Based on the amount of hyperparameters involved, this makes the processing time-consuming and costly. GridSearchCV does cross-validation in addition to grid search. The model is trained using cross-validation.  

As we all know, we divide the data into two pieces before training the model with it: train data and test data. The procedure of cross-validation divides the train data into two parts: the train data and the validation data.K-fold Cross-validation is the most common type of cross-validation. The train data is divided into k divisions using an iterative approach. One division is kept for testing and the remaining k-1 partitions are used to train the model in each iteration. In the next iteration, the next partition will be used as test data, and the remaining k-1 will be used as train data, and so on. It will record the model’s performance in each iteration and offer the average of all the results in the end. As a result, it is a time-consuming operation.  

**5.3.3]SAMPLE CODING i]![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.022.png)Text preprocessing:**  

from tqdm import \* 

class Data\_cleaner: 

`  `def Data\_Cleaning(self,text): 

#text cleaning text=re.sub(r'(\d+)',r'',text) text=text.replace(u',','') text=text.replace(u'"','') text=text.replace(u'(','') text=text.replace(u')','') text=text.replace(u'"','') text=text.replace(u':','') text=text.replace(u"'",'') text=text.replace(u"‘‘",'') text=text.replace(u"’’",'') text=text.replace(u"''",'') text=text.replace(u".",'') text=text.replace(u"\*",'') text=text.replace(u"#",'') text=text.replace(u'"','') text=text.replace(u":",'') text=text.replace(u"|",'') text=text.replace(u":","") text=text.replace(u'"',"") text = re.sub('\s+',' ', text) #Split the sentences sentences=text.split(u"।") #print(sentences) 

#Tokenizing ![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.023.png)sentences\_list=sentences tokens=[] 

`    `for each in sentences\_list: 

`            `word\_list=each.split(' ') 

`            `tokens=tokens+word\_list 

`    `#Remove token with only space     for tok in tokens: 

`        `tok=tok.strip() 

`    `#Remove hyphens in tokes 

`    `for each in tokens: 

`        `if '-' in each: 

`              `tok=each.split('-') 

`              `tokens.remove(each) 

`              `tokens.append(tok[0]) 

`              `tokens.append(tok[1]) 

`    `tokens = [i.lower() for i in tokens] 

`    `return tokens 

`  `def Text\_joiner(self,txt): 

`    `lst = ' '.join(self.Data\_Cleaning(txt))     return lst 

`  `#you need to pass the text data only   def Structured\_Data(self,sen): 

`      `lst3 = [] 

`      `for i in tqdm(range(len(sen))): 

`          `lst1 = self.Text\_joiner(sen[i])           lst3.append(lst1) 

`      `return lst3 

**ii]Vectorization:** 

from sklearn.feature\_extraction.text import TfidfVectorizer 

vec = TfidfVectorizer() ![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.024.png)

vec = vec.fit(X\_train['cleaned\_text']) 

X\_train\_bow = vec.transform(X\_train['cleaned\_text']) X\_train\_cv\_bow = vec.transform(X\_train\_cv['text']) X\_test\_bow = vec.transform(X\_test['cleaned\_text']) 

**iii]Naive Bayes Classifier:** 

from sklearn.naive\_bayes import MultinomialNB 

alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000] 

clf = MultinomialNB() 

alpha\_range = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]} clf\_Cross = GridSearchCV(clf ,param\_grid = alpha\_range ,scoring='neg\_log\_loss',cv=5,return\_train\_score=True) clf\_Cross.fit(X\_train\_bow,y\_train\_label) 

clf\_Cross.best\_params\_ 

nv\_clf = MultinomialNB(alpha=1) 

nv\_clf.fit(X\_train\_bow, y\_train\_label) 

sig\_clf = CalibratedClassifierCV(nv\_clf, method="sigmoid") sig\_clf.fit(X\_train\_bow, y\_train\_label) 

**5.3.3]SCREENSHOT** 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.025.jpeg)

**Figure 14. Importing the required Libraries** 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.026.jpeg)

**Figure 15.  Importing the Dataset**

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.027.jpeg)

**Figure 16. Visualizing the categories and Splitting of Datasets**

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.028.jpeg)

**Figure 17. Data Cleaning**

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.029.jpeg)

**Figure 18. Cleaning of X, Y Train and Test Data set**

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.030.jpeg)

**Figure 19. Training the dataset using Naïve bayes and Tuning using Hyper parameter tuning**

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.031.jpeg)

**Figure 20.  Precision, Recall, f1-score and support**

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.032.jpeg)

**Figure 21. Confusion Matrix**

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.033.jpeg)

`                                               `**Figure 22. Inferencing 5.4] SUMMARY**  

Google  Colab  is  a  cloud-based  version  of  the  Jupyter  Notebook that  provides  free computing resources.Feature extraction is used in machine learning, pattern recognition, and image processing to create derived  values (features) that are meant to be useful and  non- redundant, easing future learning and generalisation phases and, in certain situations, leading to improved human interpretations. 

Dimensionality reduction is linked to feature extraction. 

When an algorithm's input data is too vast to analyse and is suspected of being redundant (for example, the same measurement in feet and metres, or the repetitiveness of pictures provided as pixels), it can be reduced to a smaller collection of characteristics (also named a feature vector). 

Feature extraction is the process of decreasing the amount of resources needed to explain a huge amount of data. One of the primary issues with completing complicated data analysis is the large number of variables involved. 

A high number of variables necessitates a lot of memory and processing capacity, and it can also lead a classification algorithm to overfit to training examples and fail to generalise to new samples.Feature extraction is a broad phrase that refers to strategies for creating combinations of variables to get past these issues while still accurately representing the data. 

PAGE58 
`                                                       `**CHAPTER PAGE59** 

` `**6.]TESTING** 

**Data cleaning:** 

Data cleaning is the process to remove incorrect data, incomplete data and inaccurate data from the datasets, and it also replaces the missing values. There are some techniques in data cleaning 

**Handling missing values:** 

`                           `Standard values like “Not Available” or “NA” can be used to replace the missing values.  Missing values can also be filled manually but it is not recommended when that dataset is big. The attribute’s mean value can be used to replace the missing value when the data is normally distributed  wherein in the case of non-normal distribution median value of the attribute can be used. While using regression or decision tree algorithms the missing value can be replaced by the most probable  value. 

**Noisy:** 
**
`          `Noisy generally means random error or containing unnecessary data points. Here are some of the methods to handle noisy data. 

- **Binning:**  

This method is to smooth or handle noisy data. First, the data is sorted then and then the sorted values are separated and stored in the form of bins. There are three methods for smoothing data in the bin. Smoothing by bin mean method: In this method, the values in the bin are replaced by the mean value of the bin; Smoothing by bin median: In this method, the values in the bin are replaced by the median value; Smoothing by bin boundary: In this method, the using minimum and maximum values of the bin values are taken and the values are replaced by the closest boundary value. 

PAGE59 

- **Regression:**  

This is used to smooth the data and will help to handle data when unnecessary data is present. For the analysis, purpose regression helps to decide the variable which is 

suitable for our analysis**.** 

- **Clustering:**  

This is used for finding the outliers and also in grouping the data. Clustering is generally used in unsupervised learning.        

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.034.png)

**Figure 23.Data cleaning**

**Transformation of Vectors:**

Converting words to vectors, or word vectorization, is a **natural language** processing (NLP) process. The process uses language models to map words into vector space. A vector space represents each word by a vector of real numbers. It also allows words with similar meanings have similar representations.Use word embeddings as initial input for NLP downstream tasks such as text classification and sentiment analysis. 

PAGE61 

Among various word embedding technologies, in this module, we implemented three widely used methods. Two, Word2Vec and FastText, are online-training models. The other is a pretrained model, glove-wiki-gigaword-100. 

Online-training models are trained on your input data. Pretrained models are trained offline on a larger text corpus (for example, Wikipedia, Google News) that usually contains about 100 billion words. Word embedding then stays constant during word vectorization. Pretrained word models provide benefits such as reduced training time, better word vectors encoded, and improved overall performance. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.035.jpeg)

`                               `**Figure 24.Transformation of vectors Category Prediction:** 

The Text is now clean after cleaning the most recent article. Following the cleaning of the most recent article, the text is turned into vectors with tfidf and then transformed with the trained model. 

The text will then be converted to a vector using tf-idf and a trained model, then input into a neural network and turned back to text.Following the text cleaning, the article is converted to vectors using tf idf, which are then transformed using the trained model.The prediction's outcome will be shown as a numerical value.To acquire the category, the number is then inversely converted. The following done here is called inference. 

Statistical inference and modelling are crucial for interpreting data that has been influenced by chance, and hence for data scientists.These fundamental principles will be taught in this course through a stimulating case study on election predictions.This course will show you how to use inference and modelling to construct statistical procedures that make polls useful.You'll learn how to define estimates and margins of error, as well as how to utilise them to produce reasonably accurate forecasts and offer an estimate of the precision of your forecast.Once you've mastered this, you'll be able to comprehend two key concepts in data science: confidence intervals and p-values.Then you'll learn about Bayesian modelling, which will help you grasp assertions about the likelihood of a candidate prevailing. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.036.jpeg)

`  `**Figure 25.Number of categories in the dataset**

`                                                      `**CHAPTER 7 7.]RESULT ANALYSIS** 

Precision and recall were used to assess the system’s classification accuracy, i.e. the fraction of correctly categorised news documents. The model’s accuracy tells us how good it is at spotting things. There are two types of people in our world: those who are positive and those who are negative. Precision informs us about the likelihood of achieving a right positive result. The categorising of classes. The model’s recall indicates how sensitive it is to detecting the positive class.Our result analysis gives us in the form of four outcomes which are nothing but the parameter metrics such as F1 score,precision,accuracy and confusion matrix. 

**1)F1 score:** 

F1 score is the harmonic mean of precision and recall and is a better  measure than accuracy. 

The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. It is primarily used to compare the performance of two classifiers. Suppose that classifier A has a higher recall, and classifier B has higher precision. 

An F1 score is considered perfect when it's 1 , while the model is a total failure when it's 0 . Remember: All models are wrong, but some are useful. That is, all models will generate some false negatives, some false positives, and possibly both.F1 is an overall measure of a model’s accuracy that combines precision and recall, in that weird way that addition and multiplication just mix two ingredients to make a separate dish altogether. That is, a good F1 score means that you have low false positives and low false negatives, so you’re correctly identifying real threats and you are not disturbed by false alarms. An F1 score is considered perfect when it’s 1, while the model is a total failure when it’s 0. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.037.png)

**Figure 26.F1 score**

**2)Precision:** 

Precision helps when the costs of false positives are high. So let’s assume the problem involves the detection of skin cancer. If we have a model that has very low precision, then many patients will be told that they have melanoma, and that will include some misdiagnoses. Lots of extra tests and stress are at stake. When false positives are too high, those who monitor the results will learn to ignore them after being bombarded with false alarms.The ability of a classification model to identify only the relevant data points. Mathematically, precision the number of true positives divided by the number of true positives plus the number of false positives. Precision is important in machine learning because when we have an imbalanced class and we need high true positives, precision is prefered over recall. because precision has no false negative in its formula, which can impact. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.038.jpeg)

**Figure 27.Precision**

**3)Recall:** 

Recall is a metric that quantifies the number of correct positive predictions made out of all positive predictions that could have been made. Unlike precision that only comments on the correct positive predictions out of all positive predictions, recall provides an indication of missed positive predictions.Recall also gives a measure of how accurately our model is able to identify the relevant data.Recall is more important than precision when the cost of acting is low, but the opportunity cost of passing up on a candidate is high.Recall actually calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive). Applying the same understanding, we know that Recall shall be the model metric we use to select our best model when there is a high cost associated with False Negative.For instance, in fraud detection or  sick  patient  detection.  If  a  fraudulent transaction  (Actual  Positive)  is  predicted  as  non- fraudulent (Predicted Negative), the consequence can be very bad for the bank.Similarly, in sick patient detection. If a sick patient (Actual Positive) goes through the test and is predicted as not sick (Predicted Negative). The cost associated with False Negative will be extremely high if the sickness is contagious. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.039.png)

**Figure  28.Classification report of Naive Bayes**

**4)Confusion matrix** 

A confusion matrix, also known as an error matrix, is a special table structure that permits visualisation of the performance of an algorithm, often a supervised learning one, in the field of machine learning and specifically the issue of statistical classification (in unsupervised learning it is usually called a matching matrix). 

The examples in an actual class are represented by each row of the matrix, whereas the instances in a predicted class are represented by each column, or vice versa - both variations are documented in the literature. 

The term originates from the fact that it makes it simple to observe whether the system is confusing two classes (i.e. regularly mislabeling one as another) (i.e. commonly mislabeling one as another). A table of confusion (also known as a confusion matrix) is a two-row, two-column table that provides the number of false positives, false negatives, true positives, and true negatives in predictive analytics. 

`                    `This provides for a more extensive examination than simply looking at the percentage of right classifications (accuracy).If the data set is imbalanced, that is, when the number of observations in various classes varies substantially, accuracy will produce deceptive findings. 

![](Aspose.Words.90f1947b-d9ad-4daa-b341-2b1b284b1d8a.040.png)

**Figure 29.Confusion Matrix of Naive Bayes Classifier** 



|S.No |Train Accuracy |CV Accuracy |Test Accuracy |
| - | - | - | - |
|1. |99% |99% |99% |
**Table1:Accuracy of Train, Test and Cross Validation Datasets**

PAGE67 
**CHAPTER PAGE68** 

**8.]CONCLUSION** 

The rapidly expanding field of online newspapers is a rich resource that can greatly benefit from an automated classification strategy.We present a system for automatically classifying Tamil News documents in this study. This technology gives customers quick and secure access to classified news from a variety of sources. It achieves great classification accuracy of 

99%, with the possibility of a single storey being categorised into multiple categories. To solve our classification challenge, we used the Naive Bayes algorithm, which is based on a probabilistic foundation. To attain improved accuracy in Tamil news document categorization, natural language processing and additional data are critical. 

We present a system for automatically classifying Tamil News documents in this study. This technology gives cus- tomers quick and secure access to classified news from a variety of sources. It achieves great classification accuracy of 99%, with the possibility of a single storey being categorised into multiple categories. To solve our classification challenge, we used the Naive Bayes algorithm, which is based on a probabilistic foundation. To attain improved accuracy in Tamil news document categorization, natural language processing and additional data are critical. The Naive Bayes model is an excellent model for large- scale NLP tasks like news classification because it is a suitable model for large-scale NLP activities. The larger the model, the more accurate it is, but the work will take longer to complete. The Naive Bayes classifiers provide insightful outcomes in the fields of detecting sentiments and spam in text contexts .  

`                             `Varying to the fields, the need for a classifier also differentiates, since the math behind the algorithm changes. In order to be able to accurately produce correct word classes and categories, it is highly essential to define the type of problem. When we wish to employ a Naive Bayes model and process vast volumes of data, we can benefit from using NLP. When compared to other NLP models, we observed that utilizing Naive Bayes is more efficient in our investigation. With the help of the multinomial naive bayes model, we have attained a level of accuracy of 99 %. We intend to enhance and improve our framework in the near future by investigating different architectures and pre-trained models to improve classification. 

PAGE68 

**8.]FUTURE WORKS** 

Naive bayes model, it improves classification performance and computational resources. It is easy and fast to predict the class of a test data set. It also performs well in multi class prediction. When assumption of independence holds, a Naïve Bayes classifier performs better compared to other models like logistic regression and you need less training data.  

In the near future, we plan to expand and improve our framework by exploring more architectures and pre-trained models to improve classification performance and computational resources. Furthermore, we wanted to explore the effects of text preprocessing prior to training. 

Future research proposals include 

1) implementing TC experiments using additional feature types, e.g., word/character n-grams, 

skip word/character n-grams 

2) applying other ML methods such as deep learning methods, and  
3) conducting experiments on additional benchmark corporations written in Tamil.

PAGE69 

**9.] REFERENCES** 

[1].N. Rajkumar, T. S. Subashini, K. Rajan and V. Ramalingam, An Ensemble of Feature Selection with Deep Learning based Automated Tamil Document Classification Models, International Journal of Electrical Engineering and Technology, 11(9), 2020. 

2. N. Rajkumar, T. S. Subashini, K. Rajan and V. Ramalingam, An Efficient Feature Extraction with Subset Selection Model using Machine Learning Techniques for Tamil Documents Classification, International Journal of Advanced Research in Engineering and Technology, 11(11), 2020. 
2. S. Thavareesan and S. Mahesan, ”Sentiment Analysis in Tamil Texts: A Study on Machine Learning Techniques and Feature Representation,” 2019 14th Conference on Industrial and Information Systems (ICIIS),2019. 
2. R. Srinivasan and C. N. Subalalitha, ”Automated Named Entity Recognition from Tamil Documents,” 2019 IEEE 1st International Conference on Energy, Systems and Information Processing (ICESIP), 2019. 
2. M.S. Faathima Fayaza,Surangika Ranathunga,”Tamil News Clustering Using Word Embeddings”,2020 Moratuwa Engineering Research Conference (MERCon),July 2020. 
2. Bo Huang,Yang Bai,”HUB@DravidianLangTech-EACL2021:Identify and Classify Offensive Text in Multilingual Code Mixing in Social Media”,April 2021. 
2. N. Rajkumar, T. S. Subashini, K. Rajan, V. Ramalingam,”A Survey:Feature Selection and Machine Learning Methods for Tamil Text Classification,International Journal of Recent Technology and Engineering (IJRTE),May 2020. 
2. Omar Einea,Ashraf Elnagar,Ridhwan Al-Debsi,”SANAD: Single-Label Arabic News Articles Dataset for Automatic Text Categorization”,September 2019. 
2. Wongkot Sriurai,”Improving text categorization by using a topic model”,An International Journal ( ACIJ ),November 2011. 
2. Johnson Kolluri, Shaik Razia,”Text classification using Naive Bayes Classifier”,October 2020. 
PAGE71 
