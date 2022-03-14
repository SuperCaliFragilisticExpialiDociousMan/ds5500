import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def dataframe_creation():
    df = pd.read_csv('/users/shizhengyan/Desktop/Neu/DS5500/Project/new_low_risk/groupedSets/masterKey.csv')
    df['filename'] = '/users/shizhengyan/Desktop/Neu/DS5500/Project/new_low_risk/' + df['filepath'] + '/' + df['filename']
    df = df.drop('filepath', axis=1)
    for i in range(len(df)):
        if df.iloc[i,-1]=='N':
            df.iloc[i, -1]=[1,0,0,0,0,0,0,0]
            #df.iloc[i, -1] = 1
        elif df.iloc[i,-1]=='D':
            df.iloc[i, -1]=[0,1,0,0,0,0,0,0]
            #df.iloc[i, -1] = 2
        elif df.iloc[i,-1]=='G':
            df.iloc[i, -1]=[0,0,1,0,0,0,0,0]
            #df.iloc[i, -1] = 3
        elif df.iloc[i,-1]=='C':
            df.iloc[i, -1]=[0,0,0,1,0,0,0,0]
            #df.iloc[i, -1] = 4
        elif df.iloc[i,-1]=='A':
            df.iloc[i, -1]=[0,0,0,0,1,0,0,0]
            #df.iloc[i, -1] = 5
        elif df.iloc[i,-1]=='H':
            df.iloc[i, -1]=[0,0,0,0,0,1,0,0]
            #df.iloc[i, -1] = 6
        elif df.iloc[i,-1]=='M':
            df.iloc[i, -1]=[0,0,0,0,0,0,1,0]
            #df.iloc[i, -1] = 7
        elif df.iloc[i,-1]=='O':
            df.iloc[i, -1]=[0,0,0,0,0,0,0,1]
            #df.iloc[i, -1] = 8
    return df


df = dataframe_creation()
print(df)



w , h= 16,16
final_class = 8

listImg = os.listdir('/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/preprocessed_images')
string = '/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/preprocessed_images/'
list2 = list(map(lambda orig_string: string + orig_string , listImg))

fundus = []


y=[]
drop_rows=[]
for i in range(len(df)):
    location=df.iloc[i]['filename']
    #if location in list2:
    img = cv2.imread(location, 0)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = img.reshape(w, h, 1)
    fundus.append(img)
    row_index=len(fundus)-1

    t = df.iloc[i, -1]

    if  (t == [1, 0, 0, 0, 0, 0, 0, 0] and y.count(t) >= 1000) or (
            t == [0, 1, 0, 0, 0, 0, 0, 0] and y.count(t) >= 1000) or (
            t == [0, 0, 1, 0, 0, 0, 0, 0] and y.count(t) >= 1000) or (
            t == [0, 0, 0, 1, 0, 0, 0, 0] and y.count(t) >= 1000) or (
            t == [0, 0, 0, 0, 1, 0, 0, 0] and y.count(t) >= 1000) or (
            t == [0, 0, 0, 0, 0, 1, 0, 0] and y.count(t) >= 1000) or (
            t == [0, 0, 0, 0, 0, 0, 1, 0] and y.count(t) >= 1000) or (
            t == [0, 0, 0, 0, 0, 0, 0, 1] and y.count(t) >= 1000):
        drop_rows.append(row_index)
    else:
        y.append(t)



X=[]
for i in range(len(fundus)):
    if i not in drop_rows:
        X.append(fundus[i])

X = np.array(X)


print('number of N: ', y.count([1,0,0,0,0,0,0,0]))
print('number of D: ', y.count([0,1,0,0,0,0,0,0]))
print('number of G: ', y.count([0,0,1,0,0,0,0,0]))
print('number of C: ', y.count([0,0,0,1,0,0,0,0]))
print('number of A: ', y.count([0,0,0,0,1,0,0,0]))
print('number of H: ', y.count([0,0,0,0,0,1,0,0]))
print('number of M: ', y.count([0,0,0,0,0,0,1,0]))
print('number of O: ', y.count([0,0,0,0,0,0,0,1]))



y = np.array(y)
print('Label :   '+str(y.shape))


tf.config.experimental_run_functions_eagerly(True)
#tf.config.run_functions_eagerly(True)
#tf.config.run_functions_eagerly(run_eagerly)


print(1)

def our_model():
    input = Input(shape=(X.shape[1], X.shape[2], 1,))

    new_input = Input(shape=(X.shape[1], X.shape[2], 3))
    conv = Conv2D(3, kernel_size=3, padding='same',  activation='relu')(input)
    conv_input = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=new_input, input_shape=None, pooling='avg',)(conv)

    C1 = Dense(1024, activation='relu')(conv_input)
    C2 = Dense(512, activation='relu')(C1)
    C3 = Dense(256, activation='relu')(C2)
    C4 = Dense(128, activation='relu')(C3)
    C5 = Dense(64, activation='relu')(C4)
    output = Dense(final_class, activation='sigmoid')(C5)
    model = Model(inputs=input, outputs=output)
    return model


md = our_model()



print(1)

################################################
'''Train Test Split'''

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=37)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=37)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def model_training():
        model = our_model()   # function we have previously defined
        model.compile(optimizer='Adagrad',  # we can choose SGD, Adagrad, Adadelta, RMSprop ,etc
                loss='binary_crossentropy',metrics=['accuracy'])
        fit = model.fit(X_train, y_train, epochs=1,verbose=1,shuffle = True, validation_data=(X_valid,y_valid))
        return model,fit

model,fit= model_training()

test_result=model.predict(X_test.astype(np.float32))



##############################################################





METRICS = [
                'accuracy',
                tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.TruePositives()
        ]
model.compile(
                optimizer='Adam',
                loss='binary_crossentropy',
                metrics=METRICS
            )
score = model.evaluate(X, y, verbose=0)

for i in range(len(score)):
    print(model.metrics_names[i]+" : "+str(score[i]))



from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
yhat = model.predict(X_test.astype(np.float32))
print(yhat)


yhat_final=[[0,0,0,0,0,0,0,0] for i in range(len(yhat))]
for i in range(len(yhat)):
    l=yhat[i].tolist()
    j=l.index(max(l))
    yhat_final[i][j]=1
yhat_final=np.array(yhat_final)
print(yhat_final)
'''

yhat_final=[0 for i in range(len(yhat))]
for i in range(len(yhat)):
    value=round(yhat[i])
    if value<1:
        value=1
    if value>8:
        value=8
    yhat_final[i][j]=value
yhat_final=np.array(yhat_final)
print(yhat_final)
'''
report = classification_report(y_test, yhat_final,output_dict=True)
df = pd.DataFrame(report).transpose()
print(df)

correct=0
for i in range(len(y_test)):
    if str(y_test[i])==str(yhat_final[i]):
        correct+=1
print('real test accuracy: ', correct/len(y_test))

