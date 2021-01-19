# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:44:31 2020

@author: Didem Kaymaz 171307007 Makine Öğrenmesi Projesi(VİZE)
"""
#kütüphaneler yüklendi.
import pandas as pd
import numpy as np
#veri kümesi alındı. 
data = pd.read_csv('17kdata.csv',engine='python',sep = ';')#csv içine aktarmak için sep parametresi kullanıldı.
print(data)
#Veri kümesine sütun adları verildi.
data_cols = ['Tweetler', 'durum'] 
data.columns = data_cols
#Bağımlı ve Bağımsız değişkenler iloc metoduyla ayrıldı.
sentences_training = [doc for doc in data.iloc[:,0]]
classification_training = [doc for doc in data.iloc[:,1]]


import seaborn as sns
import matplotlib.pyplot as plt
#Olumlu/Olumsuz/Nötr durumların yüzdelik durumları grafiği çizildi.

plot_size = plt.rcParams["figure.figsize"] 
plt.title('Duygu Dağılımı')
plt.rcParams["figure.figsize"] = plot_size 
data.durum.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])


#Tweetler vektöre çevrildi.
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', lowercase = True)
sen_train_vector = vectorizer.fit_transform(sentences_training)
print(sen_train_vector)
 
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# Kategorik olan hedef değişkeni nümerik yapıldı.
encoder = LabelEncoder()
classification_training= encoder.fit_transform(classification_training)
from keras.utils import np_utils
#(gölge değişken oluşturuldu)
# 0 0 1=olumsuz
# 0 1 0=olumlu
# 1 0 0=nötr
y=np_utils.to_categorical(classification_training)
print(classification_training)



from sklearn.model_selection import train_test_split
#Veri seti, eğitim seti ve test seti olarak ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(sen_train_vector.toarray(), y, test_size=0.50, random_state=0)


from keras.preprocessing.text import Tokenizer #cümle,kelime,harf halinde parçalamak için kullanılır.
from keras.preprocessing.sequence import pad_sequences
token = Tokenizer()
# Her kelimenin sıklığı hesaplandı
token.fit_on_texts(sentences_training)
# Tüm kelimeler sayı dizisine dönüştürüldü.
sentences_training = token.texts_to_sequences(sentences_training)
# tüm metinler en uzun kelimeden oluşan metin(en uzun metin 52 kelimeden oluşmaktadır.) kadar sütundan oluşan bir dizi oluşturuldu.
# 52'den kısa olan metinlerin boşlukları 0 ile dolduruldu.
sentences_training = pad_sequences(sentences_training)

#Yapay sinir ağları girdi olarak niteliklerin standardize edilmiş hallerini istemektedir.
#scikit learn kütüphanesi preprocessing modülü StandardScaler sınıfını kullanarak standardize edilmiştir.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Keras kütüphanelerini indirme
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() #nesne oluşturuldu.
#Adding the input layer and first hidden layer
classifier.add(Dense(kernel_initializer = 'uniform', units = 3, input_dim=40795,  activation = 'tanh'))
#Adding second hidden layer
classifier.add(Dense(kernel_initializer = 'uniform',units = 3, activation = 'tanh'))
#Adding the Output Layer
classifier.add(Dense(kernel_initializer = 'uniform',units = 3, activation = 'softmax'))
# compile() metodu ile derlendi.
#loss fonksiyonu: gerçek değer ile tahmin edilen değer arasındaki hatayı ifade eden metrik.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

egitim=classifier.fit(X_train,y_train,validation_data=(X_test,y_test), batch_size=5, shuffle=True, verbose=1, epochs=3)
#print(X_train[0].shape) 

scores = classifier.evaluate(X_test,y_test)
print("Accuracy: ",scores[1])


# sen_test_vector = vectorizer.transform(['#Turkcell Spotify olayı olursa gecerim turkcelle'])
# #print(sen_test_vector.toarray())
# y_pred = classifier.predict(sen_test_vector)
# print(y_pred)


import seaborn as sns
#Accuracy Grafiği
from matplotlib import pyplot as plt
plt.plot(egitim.history['accuracy'])
plt.plot(egitim.history['val_accuracy'])
plt.plot(egitim.history['loss'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test','Loss'], loc='upper left')
plt.show()
