# Classification Of Tamil News Articles By Naive Bayes Model Using Natural Language Processing 



[![Google Colab](https://colab.research.google.com/img/colab_favicon_256px.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Machine learning has created a drastic impact in every sector that has integrated it into business processes such as education, healthcare, banking services and etc. The current development of Machine Learning algorithms helps to attain effective Tamil document classification. Automatic text classification aims to allocate fixed class labels to unclassified text documents. NLP problems are unclear for languages other than English. The problems may be named as Entity Extraction, OCR or classification. So in this project we are going to use Naive Bayes model to categorize the Tamil articles in an efficient way. With the help of this Naïve bayes technique, the dataset and the trained model we can achieve the desired output within the stipulated time. So that with the help of the trained model, we can achieve the desired accuracy in a higher level.



## COUNT VECTORIZER

Characters and words are incomprehensible to machines.
As a result, while dealing with text data, we must express it numerically so that the machine can understand it. Count Vectorizer is a text-to-numerical data conversion method.
Count Vectorizer tokenizes (breaks down a sentence, paragraph, or any text into words) the text and performs very basic preprocessing such as removing punctuation marks, turning all words to lowercase, and so on. A vocabulary of recognised words is built, which will subsequently be utilised to encode unseen material. The result is an encoded vector with the whole vocabulary's length and an integer count of how many times each word appears in the document.


# NAÏVE BAYES CLASSIFIER

The Naive Bayes classifier is based on the Bayes theorem’s
conditional probability principle .You use Bayesian. System Architecture of the proposed model
likelihood to reason backwards to find the events or random causes that most likely caused a specific result. These arbitrary factors in this content arrangement model will be the terms
in the archive and their frequencies. Word recurrence is the component on which a multinomial guileless Bayes classifier bases its calculation . The model will distribute the report
to the class, classification, as a result of the outcome



## Text preprocessing:
```sh
from tqdm import * 
class Data_cleaner: 
def Data_Cleaning(self,text): 

text=re.sub(r'(\d+)',r'',text) text=text.replace(u',','') text=text.replace(u'"','') text=text.replace(u'(','') text=text.replace(u')','') text=text.replace(u'"','') text=text.replace(u':','') text=text.replace(u"'",'') text=text.replace(u"‘‘",'') text=text.replace(u"’’",'') text=text.replace(u"''",'') text=text.replace(u".",'') text=text.replace(u"*",'') text=text.replace(u"#",'') text=text.replace(u'"','') text=text.replace(u":",'') text=text.replace(u"|",'') text=text.replace(u":","") text=text.replace(u'"',"") text = re.sub('\s+',' ', text) #Split the sentences sentences=text.split(u"।") 
```

## Conclusion 
The Naïve bayes model is a good model to do large scale NLP tasks such as news classification. The larger the model gives higher accuracy, but it will take more time to complete the task. Using NLP gives us advantages when we want to use a Naïve bayes model and process large amounts of data. In this study we found that using Naïve bayes is more efficient when comparing with other models with NLP. So that the accuracy level we have achieved with the help of the multinomial naïve bayes model is 96.9%.





## License - MIT
Copyright (c) 2021 Balamurugan S

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.








