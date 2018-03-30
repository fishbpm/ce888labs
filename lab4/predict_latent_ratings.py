# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:37:41 2018

@author: Fish
"""

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("jester-data-1.csv", header=None)

#data = data.reindex(columns=[data.columns.tolist() + ['split']], fill_value='') 
#data.loc[:2500,'split'] = '99'

data = df.copy() #take a copy as we need the original values for validation
data.drop(data.columns[0], axis=1, inplace=True) #1st column is irrelevant to exercise

users = data.shape[0]
jokes = data.shape[1]
cycles = 5#15
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
        if(iteration%1 == 0):#000 == 0 ):
            print(mse)
    return error

t0 = time.time()
error = sgd(cycles)#000)
print(time.time() - t0,"secs to run", cycles, "iterations")
#predictions = latent_user_preferences.dot(latent_item_features.T)
#print(predictions[:10][:10])

plotdata1 = pd.DataFrame(np.vstack((np.arange(np.array(error).shape[0]), error)).T, columns=['cycles', 'MSE'])
plt.figure()
sb.lmplot(plotdata1.columns[0], plotdata1.columns[1], data=plotdata1, fit_reg=False)

plt.savefig("convergence.png",bbox_inches='tight')
plt.savefig("convergence.pdf",bbox_inches='tight')

predictions = latent_user_preferences.dot(latent_item_features.T)

error = []
for user in range(users):
    for joke in range(jokes):
        #if this was a salient data point in the original sample
        if df.iloc[user, joke + 1] != 99:
            #and we used it for validation
            if data.iloc[user, joke] == 99: 
                #then compare predicted rating to original actual rating
                err = predictions[user, joke] - df.iloc[user, joke + 1]
                error.append(err)
mse = (np.array(error) ** 2).mean()
print(mse, "error across", len(error), "original ratings")







