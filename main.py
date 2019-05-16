
# coding: utf-8

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Dense, Input
from keras.optimizers import Adam
import os, cv2
import numpy as np
import glob
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from keras.applications import Xception

# fix random seed

np.random.seed(seed=2018)
tf.set_random_seed(32)

# load dataset info

labels=pd.read_csv("train.csv").set_index('Id')
labels['Target']=[[int(i) for i in s.split()] for s in labels['Target']]
colors=['red','green','blue']

Id=[]
for i in range(0,31072):
    a=str(labels.Id[i])
    Id.append(a)

for i in range(0,31702):
    flags=cv2.IMREAD_GRAYSCALE
    img=np.stack([cv2.imread(os.path.join('trainfile', Id[i]+'_'+color+'.png'), flags) for color in colors],-1)
    np.save('AJImage/'+Id[i],img)

folderss=glob.glob('AJImage')
imglists=[]
for folder in folderss:
    for f in glob.glob(folder+'/*.npy'):
        imglists.append(f)

imglists.sort()

IMAGE_DIMS=(299,299,3)
data=[]
for files in imglists:
    img=np.load(files)
    img=cv2.resize(img,(IMAGE_DIMS[1],IMAGE_DIMS[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    data.append(img)

data=np.array(data)

# split data into train, test

(trainX, testX, trainY, testY)=train_test_split(data, labels.Target, test_size=0.15, random_state=42)

mlb=MultiLabelBinarizer()
trainYm=mlb.fit_transform(trainY)
nlb=MultiLabelBinarizer()
testYn=nlb.fit_transform(testY)

# load predicted dataset info

ss=pd.read_csv('sample_submission.csv')

pId=[]
for i in range(0,11702):
    a=str(ss.Id[i])
    pId.append(a)

for i in range(0,11702):
    flags=cv2.IMREAD_GRAYSCALE
    img=np.stack([cv2.imread(os.path.join('testfile', pId[i]+'_'+color+'.png'), flags) for color in colors],-1)
    np.save('AJTEImage/'+pId[i],img)

predict_f=glob.glob('AJTEImage')
pimglist=[]
for folder in predict_f:
    for f in glob.glob(folder+'/*.npy'):
        pimglist.append(f)
        
pimglist.sort()

pdata=[]
for files in pimglist:
    img=np.load(files)
    img=cv2.resize(img,(IMAGE_DIMS[1],IMAGE_DIMS[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    pdata.append(img)

pdata=np.array(pdata)

# create model

def createmodel(inputshape,n_classes):
    inp_mask=Input(shape=inputshape)
    pretrain_model=Xception(include_top=False,weights='imagenet',pooling='max')
    pretrain_model.name='xception_image'
    x=pretrain_model(inp_mask)
    out=Dense(n_classes,activation='sigmoid')(x)
    model=Model(inputs=[inp_mask],outputs=[out])
    return model

model=createmodel(inputshape=(299,299,3),n_classes=28)

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc',f1])

aug = ImageDataGenerator(rotation_range=180, width_shift_range=0.1, height_shift_range=0.1, shear_range=20, zoom_range=[0.8, 1.2], horizontal_flip=True, vertical_flip=True, fill_mode='reflect')

model.fit_generator(aug.flow(trainX, trainYm, batch_size=16), steps_per_epoch=len(trainX)/16, epochs=25,validation_data=(testX, testYn), workers=20, verbose=1)

# a TTA wrapper for keras model with a predicted method

class TTA_ModelWrapper():
    def __init__(self, model):
        self.model=model
        self.gene=datagen=ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=20,
        zoom_range=[0.8,1.2],
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True)
    
    def predict_tta(self, X, aug_times=16):
        pred=[]
        for x_i in X:
            sum_p=0
            for i, d in enumerate(self.gene.flow(x_i[np.newaxis], batch_size=1)):
                if i>=aug_times:
                    break
                p=self.model.predict(d)[0]
                sum_p+=p
            pred.append(sum_p/aug_times)
        return np.array(pred)

model=TTA_ModelWrapper(model)
py=model.predict_tta(pdata,aug_times=16)

# find the threshold for each class

datath=[]
for files in imglists[26411:]:
    img=np.load(files)
    img=cv2.resize(img,(IMAGE_DIMS[1],IMAGE_DIMS[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    datath.append(img)

labels.Target[26411:]=np.array(labels.Target[26411:])
testYnth=nlb.fit_transform(labels.Target[26411:])
pred_metrix=model.predict_tta(datath,aug_times=16)


def f1_np(y_pred, y_true, threshold=0.5):
    '''numpy f1 metric'''
    y_pred = (y_pred>threshold).astype(int)
    TP = (y_pred*y_true).sum(1)
    prec = TP/(y_pred.sum(1)+1e-7)
    rec = TP/(y_true.sum(1)+1e-7)
    res = 2*prec*rec/(prec+rec+1e-7)
    return res.mean()

def f1_n(y_pred, y_true, thresh, n, default=0.5):
    '''partial f1 function for index n'''
    threshold = default * np.ones(y_pred.shape[1])
    threshold[n]=thresh
    return f1_np(y_pred, y_true, threshold)

def find_thresh(y_pred, y_true):
    '''brute force thresh finder'''
    ths = []
    for i in range(y_pred.shape[1]):
        aux = []
        for th in np.linspace(0,1,100):
            aux += [f1_n(y_pred, y_true, th, i)]
        ths += [np.array(aux).argmax()/100]
    return np.array(ths)


ths = find_thresh(pred_metrix, testYnth)
print(ths)

# create submission

y=[]
for x in py: 
    l=np.arange(28)[x>=ths]
    y.append(l)

ss['Predicted']=y

x=[]
for i in range(0,11702):
    x.append('')

for i in range(0,11702):
    for y in ss.Predicted[i]:
        x[i]+=' '+str(y)

Y=[]
for i in range(0,11702):  
    Y.append(x[i].strip())    

ss.Predicted=Y
ss.to_csv('submission.csv',index=False)

