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
    df4 = pd.read_csv('/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/full_df.csv')
    df4['filename'] = '/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/preprocessed_images/' + df4['filename']
    df4['Left-Fundus'] = '/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/preprocessed_images/' + df4[
        'Left-Fundus']
    df4['Right-Fundus'] = '/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/preprocessed_images/' + df4[
        'Right-Fundus']
    df4['Line'] = df4['Left-Diagnostic Keywords'] + ' | ' + df4['Right-Diagnostic Keywords']
    #df4 = df4.drop(['filepath', 'target'], axis=1)
    df4 = df4.drop('filepath', axis=1)
    return df4



df = dataframe_creation()
print(type(df.iloc[1,7]))
print(type(df.iloc[1,8]))
print(type(df.iloc[0:5,7]))
print(df.iloc[1,7])
print(df.iloc[1,8])

print(df['Patient Sex'].unique())



w , h= 16,16
final_class = 8

listImg = os.listdir('/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/preprocessed_images')
string = '/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/preprocessed_images/'
list2 = list(map(lambda orig_string: string + orig_string , listImg))
indexify =[]
for i in df.index:
    if df.iloc[i]['Left-Fundus'] in list2 and df.iloc[i]['Right-Fundus'] in list2:
        continue
    else:
        indexify.append(i)

print(df.shape)
df = df.drop(indexify)
print(df.shape)
print('indexify',indexify)
print('ID',df.loc[:,'ID'])


left_fundus = []
for location in tqdm(df.iloc[:]['Left-Fundus']):
    img = cv2.imread(location,0)
    img = cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
    img = img.reshape(w,h,1)
    left_fundus.append(img)
right_fundus = []
for location in tqdm(df.iloc[:]['Right-Fundus']):
    img = cv2.imread(location,0)
    img = cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
    img = img.reshape(w,h,1)
    right_fundus.append(img)
X1 = np.array(left_fundus)
X2 = np.array(right_fundus)


#y = np.array(df.iloc[:][['N','D','G','C','A','H','M','O']])
#y = np.array(y)
y=[]
for i in range(len(df)):
    target=df.iloc[i,16]
    target=target[1:-1]
    target=target.strip().split(',')
    t=[]
    for j in target:
        t.append(int(j))
    y.append(t)


y = np.array(y)
print('Label :   '+str(y.shape))


tf.config.experimental_run_functions_eagerly(True)
#tf.config.run_functions_eagerly(True)
#tf.config.run_functions_eagerly(run_eagerly)


print(1)

def our_model():
    inp1 = Input(shape=(X1.shape[1], X1.shape[2], 1,))
    inp2 = Input(shape=(X2.shape[1], X2.shape[2], 1,))
    new_input = Input(shape=(X1.shape[1], X1.shape[2], 3))
    conv1 = Conv2D(3, kernel_size=3, padding='same', activation='relu')(inp1)
    i1 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=new_input,
                                        input_shape=None,
                                        pooling='avg',
                                        )(conv1)
    conv2 = Conv2D(3, kernel_size=3, padding='same', activation='relu')(inp2)
    i2 = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_tensor=new_input,
                                          input_shape=None,
                                          pooling='avg',

                                          )(conv2)
    merge = concatenate([i1, i2])
    class1 = Dense(1024, activation='relu')(merge)
    class1 = Dense(512, activation='relu')(class1)
    class1 = Dense(256, activation='relu')(class1)
    class1 = Dense(128, activation='relu')(class1)
    class1 = Dense(64, activation='relu')(class1)
    output = Dense(final_class, activation='sigmoid')(class1)
    model = Model(inputs=[inp1, inp2], outputs=output)
    return model


md = our_model()

plot_model(md, to_file='Hybrid_neural_network.png')

print(1)

################################################
'''Train Test Split'''

X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(X1,X2, y, test_size=0.2, random_state=42)

X_train1, X_valid1, X_train2, X_valid2, y_train, y_valid = train_test_split(X_train1,X_train2, y_train, test_size=0.2, random_state=42)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def LR_verify():
        model = our_model()
        METRICS = [
                'accuracy'
        ]
        model.compile(
                optimizer='Adam',
                loss='binary_crossentropy',
                metrics=METRICS
            )
        history = model.fit([X_train1,X_train2], y_train, epochs=1,verbose=1,shuffle = True, validation_data=([X_valid1,X_valid2],y_valid))
        return model,history
model,history= LR_verify()

test_result=model.predict([X_test1.astype(np.float32),X_test2.astype(np.float32)])



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
score = model.evaluate([X1,X2], y, verbose=0)

for i in range(len(score)):
    print(model.metrics_names[i]+" : "+str(score[i]))



