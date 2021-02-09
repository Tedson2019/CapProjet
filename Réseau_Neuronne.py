# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:00:32 2021

@author: samba
"""

import numpy as np 
import pandas as pd
import tensorflow as tf
from scipy import stats
from tensorflow.keras.utils import plot_model
import sklearn.preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt


## Classe pour encoder les variables catégorielles
class CategoricalEncoder():
    def __init__(self):
        self._classes = {}
        self.size = 0

    def fit(self, X):
        uniques = np.unique(X)  #sort
        for u in uniques:
            if not u in self._classes:
                self._classes[u] = self.size
                self.size += 1
    
    def transform(self, X, unknown_category=True):
        result = np.full(X.shape, self.size, dtype=np.int32)  #full cree tab de longeur dimensions du tab X remplis de size
        for i in range(len(X)):  #range : sequence de 0 à len(X)
            try:
                result[i] = self._classes[X[i]]
            except KeyError:
                if unknown_category:
                    result[i] = self.size
                else:
                    raise KeyError
        return result
    
    def fit_transform(self, X, unknown_category=True):
        self.fit(X)
        return self.transform(X, unknown_category=unknown_category)

class ReproductionErrorLayer(tf.keras.layers.Layer):
    def __init__(self, loss, **kwargs):
        self.output_dim = 1
        self.loss = loss
        super(ReproductionErrorLayer, self).__init__(**kwargs)

    def build(self, input_shape):       
        
        super(ReproductionErrorLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        result = None
        assert isinstance(x, list)

        result = self.loss(x[0], x[1])
        
        return tf.reshape(result, shape=[-1, 1])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return (input_shape[0][0], self.output_dim)
    
    def get_config(self):
        config = {
            'loss': self.loss
        }
        base_config = super(ReproductionErrorLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


### renvoie 0 si température normale et 1 sinon
def label_to_int(label):
    if label =="Normale":
        return 0
    return 1
 
#### On appelle le dataset transformée et on traite les données en fonction de leur type 
def load_data(path ,train = False):
    
    dataset = pd.read_csv(path)
    inp = dict()
    out = dict()
    labels = None
    encode = CategoricalEncoder()
    
    for key in dataset.columns:
        if key in ["conectiondeviced"]:
            dataset[key] = encode.fit_transform(dataset[key].values)
            inp[key] = dataset[key].values.astype(np.float32)
        elif key in ["temperature","time"]:
            inp[key] = dataset[key].values.astype(np.float32)
        elif "target" in key:
            labels = dataset[key].map(label_to_int).values
            continue
        else :
            pass
        if train :
            out[key+ '-output'] = inp[key]
    if train :
        return inp , out , labels
    else:
        return inp , labels
## cette fonction nous renvoie un dictionnaire ayant pour clefs les différentes variables  du dataset et pour valeurs les mesures capturées par les capteurs 


#### Création du modèle d'entrainement ####
def create_training_model(variables):
    inputs = []
    tensors = []
    
    for key in variables:
        inp = None
        x = None 
        
        ### Catégorièle ###
        
        if key in ["conectiondeviced"] :
            inp = tf.keras.Input(shape=(1,),name=key)
            z = tf.keras.layers.Embedding(30,2,input_length=1)(inp)
            x = tf.keras.layers.Flatten()(z)
            
            
        
        #### Numérique ####
        else:
            inp = tf.keras.Input(shape=(1,),name=key)
            x = tf.keras.layers.Dense(2,activation=tf.nn.relu)(inp) 
        inputs.append(inp)
        tensors.append(x)
        
        ### On regroupe toutes les entrées ###
        
    encoder = tf.keras.layers.Concatenate()(tensors)
        
        ## On définit la partie centrale de l'autoencodeur ##
        
    Neuronal_Layer_1 = tf.keras.layers.Dense(10,activation = tf.nn.relu)(encoder)
    Neuronal_layer_2 = tf.keras.layers.Dense(15, activation = tf.nn.relu)(Neuronal_Layer_1)
    decoder = tf.keras.layers.Dense(20, activation = tf.nn.relu)(Neuronal_layer_2)
        
    losses = {}
    outputs = []
        
        ### On définit la partie décodée et les pertes spécifiques pour chaque type d'entrée 
    for key in variables: 
        loss = None
        x = None 
        if key in ["conectiondeviced"]:
            loss = tf.keras.losses.sparse_categorical_crossentropy
            x = tf.keras.layers.Dense(10, activation = tf.nn.softmax , name = key+"-output")(decoder)
               # Numeric
        else: 
            loss = tf.keras.losses.mean_squared_error 
            x = tf.keras.layers.Dense(1, activation = tf.nn.sigmoid,name=key+"-output")(decoder) 
        losses[key+"-output"]=loss
        outputs.append(x)
    return tf.keras.Model(inputs, outputs), losses

def create_inference_model(trained_model, losses, data):
    #On intègre les fonctions de perte dans le modèle 
    loss_outs = []
    for key in losses:
        in_name = key.replace("-output", "")
        layer = ReproductionErrorLayer(losses[key])([trained_model.get_layer(in_name).output, trained_model.get_layer(key).output])
        loss_outs.append(layer)

    # On Construit un modèle temporaire pour calibrer chaque perte 

    tmp = tf.keras.Model(trained_model.inputs, loss_outs)
    error = tmp.predict(data, batch_size=1024)#♣ 1024
    scalers = []
    for i in range(len(error)):
        # On calcule les paramètres utiles pour l'étalonnage
        params = np.median(error[i]),np.std(error[i])
        scalers.append(tf.keras.layers.Lambda(loss_scaler(params))(tmp.outputs[i]))


    return tf.keras.Model(tmp.inputs, tf.keras.layers.Add()(scalers))

def loss_scaler(params):
    def fn(x):
        scaler = sklearn.preprocessing.StandardScaler()
        try:
            x = scaler.fit_transform(x)
        except:
            pass
        return x
    return fn

def train_model(model, losses, data):
    model.compile(loss=losses, optimizer='adam')
    plot_model(model, to_file='autoencoder.png', show_shapes=True)
    x, y, _ = data
    model.fit(x, y, verbose=2, batch_size=1024, epochs=1000, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, min_delta=0.0001, restore_best_weights=True)])
    inf_model = create_inference_model(model, losses, x)

    return inf_model

# def find_threshold(normal_scores, anormal_scores):
#     # On calcule La Matrice de Confusion du Model grace valeur réelle et la valeur prédite
#     # On trace aussi la courbe ROC pour montrer la précition du modèle 
#     try: 
#         Matrix = sklearn.metrics.confusion_matrix(normal_scores, anormal_scores)
#     except :
#         print("erreur")
#     # print("Reussite",  (Matrix[0][0]+Matrix[1][1])/len(normal_scores) * 100,"%")
#     # fpr, tpr, threshold = sklearn.metrics.roc_curve(normal_scores, anormal_scores)
#     # roc_auc = sklearn.metrics.auc(fpr, tpr)
#     # plt.title('Courbe ROC pl')
#     # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#     # plt.legend(loc = 'lower right')
#     # plt.plot([0, 1], [0, 1],'r--')
#     # plt.xlim([0, 1])
#     # plt.ylim([0, 1])
#     # plt.ylabel('True Positive Rate')
#     # plt.xlabel('False Positive Rate')
#     # plt.show()  
#     # return 0

train_data , out , labels  = load_data(r'C:\Users\samba\OneDrive\Bureau\PFE_Réseau_Neuronne\new_dataset.csv', train=True)
# model, losses = create_training_model( train_data)
# print(losses)
# #model.summary()
# model = train_model(model, losses, train_data)
# #print(" Saving our Auto Encoder Neuronal Network in AutoEncoder_unsw-nb15.h5")
# #model.save('AutoEncoder_unsw-nb15.h5')


# # =============================================================================
# # Création du meme modèle 
# # model = keras.models.load_model('AutoEncoder_unsw-nb15.h5')
# # =============================================================================


# test_data, labels = load_data(r'C:\Users\samba\OneDrive\Bureau\PFE_Réseau_Neuronne\evaluate.csv',train=False)
# scores = model.predict(test_data, batch_size=1) # batch_size=4096

# normal_ids = np.where(labels == 0)
# anormal_ids = np.where(labels == 1)

# liste1 = []
# liste2 = []
# liste3 = []


# m = scores[anormal_ids]
# n = scores[normal_ids]

# for k in range(len(m)):
#     liste1.append(m[k][0])
# A = np.asarray(liste1)

# for k in range(len(n)):
#     liste2.append(n[k][0])
# B = np.asarray(liste2)

# C = np.concatenate((A,B))

# for k in range(len(scores)):
#     liste3.append(scores[k][0])

# D = np.asarray(liste3)

# Matrix = sklearn.metrics.confusion_matrix(C,D )

