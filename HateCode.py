import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import string
import nltk
from sklearn import metrics
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
from sklearn.metrics import confusion_matrix

stopword=set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

def clean_text(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub('\[.*?\]', '', sentence)
    sentence = re.sub('https?://\S+|www\.\S+', '', sentence)
    sentence = re.sub('<.*?>+', '', sentence)
    sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
    sentence = re.sub('\n', '', sentence)
    sentence = re.sub('\w*\d\w*', '', sentence)
    sentence = [word for word in sentence.split(' ') if word not in stopword]
    sentence=" ".join(sentence)
    sentence = [stemmer.stem(word) for word in sentence.split(' ')]
    sentence=" ".join(sentence)
    return sentence

def classifier_evaluation(y_pred, y_test):
    # Confusion Matrix & Classification Report
    fig, ax = plt.subplots()
    confusion_matrix = pd.crosstab(y_pred, y_test, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, cmap = 'Blues')
    acc=metrics.accuracy_score(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(fig)
    
    st.write("Accuracy :",acc)
    st.text('Model Report:\n ' + classification_report(y_pred, y_test))

    
    
df = pd.read_csv('data/labeled_data.csv')

# Data Cleaning
del df['Unnamed: 0']
del df['count']
del df['hate_speech']
del df['offensive_language']
del df['neither']

clean_tweets = []

for index, row in df.iterrows():
    temp_sentence = row['tweet']
    temp_sentence = clean_text(temp_sentence)
    clean_tweets.append(temp_sentence)

df['clean_tweet'] = clean_tweets   

new_class = []

for index, row in df.iterrows():
    temp_class = row['class']
    if temp_class > 1:
        temp_class = 1
    new_class.append(temp_class)
    
df['binary_class'] = new_class

df.rename(columns={'class':'trinary_class'}, inplace = True)



st.header('**Hate tweets Classification**')
st.write('---')

menu = st.sidebar.selectbox("Select a Function", ("Dataset Report", "Tweet Classification Models"))

if menu == "Dataset Report":
    pr = ProfileReport(df, explorative=True)
    st.header('*Pandas Profiling Report*')
    st_profile_report(pr)
        
if menu == "Tweet Classification Models":  
    st.header('*Model Evaluation*')
    # Train & Test Data Split
    x = df['clean_tweet'].astype(str)
    y1 = df['binary_class'].astype(str)
    y2 = df['trinary_class'].astype(str)
    
    x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x, y1, y2, test_size = 0.2, random_state = 42)
    
    # Vectorize Tweets using TF-IDF
    #from sklearn.feature_extraction.text import CountVectorizer
    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(df['clean_tweet'].astype(str))
    x_train_tfidf = tfidf_vect.transform(x_train)
    x_test_tfidf = tfidf_vect.transform(x_test)
    

    if st.checkbox('Evaluate The SVM-Binary Classification Model (Hate, Non-Hate)'):
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        svm_binary = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
        svm_binary.fit(x_train_tfidf, y1_train)

        # predict the labels on validation dataset
        y1_pred = svm_binary.predict(x_test_tfidf)
        # Classifier Evaluation
        classifier_evaluation(y1_pred, y1_test)
        
        #User Input
        st.write("""##### Try it out yourself!""")
        binary_text = st.text_area("Classify Using The SVM- Binary Model:", "Enter Text")  
        #Clean the Text
        binary_text = clean_text(binary_text)
        
        if st.checkbox('Apply SVM-Binary Model'):
            # Preparation for Classifying User Input
            binary_model = Pipeline([('vectorizer', tfidf_vect), ('classifier', svm_binary)])
            
            
            # Generate Result
            result = binary_model.predict([binary_text])
            
            if result.astype(int) == 0:
                result_text = "Hate Speech"
            else:
                result_text = "Non-Hate Speech"
                
            st.write(" ##### Result: ", result_text)
           
            # Interpretation of Result
            st.write("""#### Result Interpretation:""")
            binary_model.predict_proba([binary_text])
            binary_explainer = LimeTextExplainer(class_names={"Hate":0, "Non-Hate":1})
            max_features = x_train.str.split().map(lambda x: len(x)).max()
            
            
            random.seed(13)
            idx = random.randint(0, len(x_test))
            
            bin_exp = binary_explainer.explain_instance(binary_text, binary_model.predict_proba, num_features=max_features)
            
            components.html(bin_exp.as_html(), height=800)
                
    

    if st.checkbox('Evaluate The Logistic Regression-Trinary Classification Model (Hate, Offensive, Neither)'):
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        lr_trinary = LogisticRegression(C=0.05,solver='lbfgs',multi_class='auto')
        lr_trinary.fit(x_train_tfidf, y2_train)

        # predict the labels on validation dataset
        y2_predlr = lr_trinary.predict(x_test_tfidf)
        # Classifier Evaluation
        classifier_evaluation(y2_predlr, y2_test)
        
        #User Input
        st.write("""##### Try it out yourself!""")
        #User Input
        trinary_text = st.text_area("Classify Using The LR-Trinary Model:", "Enter Text")  
        #Clean the Text
        trinary_text = clean_text(trinary_text)
        
        if st.checkbox('Apply LR-Trinary Model'):
            # Preparation for Classifying User Input
            trinary_modellr = Pipeline([('vectorizer', tfidf_vect), ('classifier', lr_trinary)])
            
            # Generate Result
            result = trinary_modellr.predict([trinary_text])
            
            if result.astype(int) == 0:
                result_text = "Hate Speech"
            elif result.astype(int) == 1:
                result_text = "Offensive Language"
            else:
                result_text = "Neither Hate Nor Offensive"
                
            st.write(" ##### Result: ", result_text)
    
    if st.checkbox('Evaluate The Naive bayes-Trinary Classification Model (Hate, Offensive, Neither)'):
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        NB_trinary=MultinomialNB(alpha=1.7)
        NB_trinary.fit(x_train_tfidf, y2_train)

        # predict the labels on validation dataset
        y2_predNB = NB_trinary.predict(x_test_tfidf)
        # Classifier Evaluation
        classifier_evaluation(y2_predNB, y2_test)
        
        #User Input
        st.write("""##### Try it out yourself!""")
        #User Input
        trinary_text = st.text_area("Classify Using The NB-Trinary Model:", "Enter Text")  
        #Clean the Text
        trinary_text = clean_text(trinary_text)
        
        if st.checkbox('Apply NB-Trinary Model'):
            # Preparation for Classifying User Input
            trinary_modelNB = Pipeline([('vectorizer', tfidf_vect), ('classifier', NB_trinary)])
            
            # Generate Result
            result = trinary_modelNB.predict([trinary_text])
            
            if result.astype(int) == 0:
                result_text = "Hate Speech"
            elif result.astype(int) == 1:
                result_text = "Offensive Language"
            else:
                result_text = "Neither Hate Nor Offensive"
                
            st.write(" ##### Result: ", result_text)
           
            
    

    
    
