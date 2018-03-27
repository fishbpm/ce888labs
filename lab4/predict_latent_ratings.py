# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:37:41 2018

@author: Fish
"""

import numpy as np
import pandas as pd
import random
import time

df = pd.read_csv("jester-data-1.csv", header=None)

#data = data.reindex(columns=[data.columns.tolist() + ['split']], fill_value='') 
#data.loc[:2500,'split'] = '99'

data = df.copy() #take a copy as we need the original values for validation
data.drop(data.columns[0], axis=1, inplace=True) #1st column is irrelevant to exercise

users = data.shape[0]
jokes = data.shape[1]
#now randomly remove 2500 ratiings
#only valid ratings (<> 99) are removed, until we reach the target 2500
i = 0
while i < 2500:
    user = random.sample(range(0, users), 1)[0]
    joke = random.sample(range(0, jokes), 1)[0]
    if data.iloc[user, joke] != 99:
        data.iloc[user, joke] = 99
        i += 1

n_features = 2

user_ratings = data.values
latent_user_preferences = np.random.random((user_ratings.shape[0], n_features))
latent_item_features = np.random.random((user_ratings.shape[1], n_features))
    

def predict_rating(user_id,item_id):
    """ Predict a rating given a user_id and an item_id.
    """
    user_preference = latent_user_preferences[user_id]
    item_preference = latent_item_features[item_id]
    return user_preference.dot(item_preference)


def train(user_id, item_id, rating, alpha = 0.0001):
    
    #print(item_id)
    prediction_rating = predict_rating(user_id, item_id)
    err =  prediction_rating - rating
    #print(err)
    user_pref_values = latent_user_preferences[user_id][:]
    latent_user_preferences[user_id] -= alpha * err * latent_item_features[item_id]
    latent_item_features[item_id] -= alpha * err * user_pref_values
    return err
    
def sgd(iterations):
    """ Iterate over all users and all items and train for 
        a certain number of iterations
    """
    for iteration in range(0,iterations):
        error = []
        for user_id in range(0,latent_user_preferences.shape[0]):
            for item_id in range(0,latent_item_features.shape[0]):
                rating = user_ratings[user_id][item_id]
                if rating != 99:
                    err = train(user_id, item_id, rating)
                    error.append(err)
        mse = (np.array(error) ** 2).mean()   
#        if(iteration%10 == 0):#000 == 0 ):
#            print(mse)
        print(mse)
    return

t0 = time.time()
sgd(3)#000)
print(time.time() - t0)
#predictions = latent_user_preferences.dot(latent_item_features.T)
#print(predictions[:10][:10])