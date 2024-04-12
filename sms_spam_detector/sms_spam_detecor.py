import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
# import chardet
# with open("./spam.csv","rb") as x:
#     sonuc=chardet.detect(x.read())
# print(sonuc)
pd.set_option("display.max_columns",None)
data=pd.read_csv("./spam.csv",encoding='Windows-1252')
veri=data.copy()

veri.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)

veri.rename(columns={"v1":"Etiket","v2":"Sms"},inplace=True)
print(veri)
print(veri.groupby("Etiket").count())
print(veri.describe())

veri.drop_duplicates(inplace=True) #tekrarlı verileri sildik
print(veri.describe())
print(veri.isnull().sum())

veri["karakter sayısı"]=veri["Sms"].apply(len)

# veri.hist(column="karakter sayısı",by="Etiket",bins=50)
# plt.show()

veri.Etiket=[1 if kod=="spam" else 0 for kod in veri.Etiket]


mesaj=re.sub("[^a-zA-Z]"," ",veri.Sms[0])


def harfler(cumle):
    yer=re.compile("[^a-zA-Z]")
    return re.sub(yer," ",cumle)

durdurma=stopwords.words("english")



spam=[]
ham=[]
tumcumleler=[]

for i in range(len(veri.Sms.values)):
    r1=veri["Sms"].values[i]
    r2=veri["Etiket"].values[i]

    temizcumle=[]
    cumleler=harfler(r1)
    cumleler=cumleler.lower()
    for kelimeler in cumleler.split():
        temizcumle.append(kelimeler)
        if r2==1:
            spam.append(cumleler)
        else:
            ham.append(cumleler)
    tumcumleler.append(" ".join(temizcumle))
veri["Yeni Sms"]=tumcumleler

veri.drop(columns=["Sms","karakter sayısı"],axis=1,inplace=True)
print(veri)

cv=CountVectorizer()
x=cv.fit_transform(veri["Yeni Sms"]).toarray()
y=veri["Etiket"]
X=x
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
for i in np.arange(0.0,1.1,0.1):
    model=MultinomialNB(alpha=i)
    model.fit(X_train,y_train)
    tahmin=model.predict(X_test)
    acs=accuracy_score(y_test,tahmin)
    print(i,"  :   ",acs*100)
