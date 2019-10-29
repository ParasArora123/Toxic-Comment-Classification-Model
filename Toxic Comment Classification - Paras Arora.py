#!/usr/bin/env python
# coding: utf-8

# # Analyzing Toxic Comments Online
# 
# For this project, I will be classifying comments (found in a kaggle dataset) as 'toxic', 'severe toxic', 'obscene', 'threat', 'insult', and or 'identity hate' (or none of these). Any blog or site who has access to my model should be able to run the algorithm on their site to clear comments they deem innappropriate based on the categories above.
# 
# In this notebook, I walk through each step from the basic preprocessing to the more complex models I finally build out. Below I start by importing all libraries I need for the project:

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# For Natural Language Processing: 

# In[3]:


import re  
import nltk

nltk.download('stopwords') # Downloading the words that are not relevant to exclude them from the data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # for stemming -- not sure if I will go through with this 


# In[6]:


import sys # In order to import tensorflow 
sys.executable

from keras.preprocessing.text import Tokenizer # For tokenization 
from keras.preprocessing.sequence import pad_sequences # To make all the sentences equal in length 
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation 
from keras import initializers, regularizers, constraints, optimizers, layers 
from keras.layers import Bidirectional, GlobalMaxPool1D 
from keras.models import Model


# ## Analyzing the Dataset
# 
# Below is the training set (all 95851 comments) that I am using to run my model on. As you can see each section has multiple different columns that can be classified as either a '0' or a '1'. 

# In[7]:


train_toxic = pd.read_csv("data/train_toxic.csv")
train_toxic


# ### Checking for Null Values
# 
# Checking if any of the sections are null / I need to fill in and or change any data (it turns out I don't!) 

# In[8]:


train_toxic.info()


# ### Counting Classifications
# 
# Now I want to count how many comments are classified under each category. (e.g. how many 'obscene' examples are there) -- I use the .sum() function 

# In[9]:


count = pd.DataFrame(columns=['Toxicity count', 'Severe_toxic count', 'Obscene count',
                             'Threat count', 'Insult count', 'Identity_hate count'])

count = count.append({'Toxicity count': train_toxic.toxic.sum(), 
                    'Severe_toxic count': train_toxic.severe_toxic.sum(),
                    'Obscene count': train_toxic.obscene.sum(),
                    'Threat count': train_toxic.threat.sum(),
                    'Insult count': train_toxic.insult.sum(),
                    'Identity_hate count': train_toxic.identity_hate.sum()}, 
                    ignore_index=True)
count


# ### Finding Correlation 
# 
# Trying to find the correlation between each of the target categories...

# In[10]:


corr = train_toxic.corr()
corr


# ### Investigating Individual Comments
# 
# Here I investigate indvidual comments and attempt to on the surface look at the difference between threats, insults, and identity_hate by looking at comments from each section that contains '1' for that section. 

# In[11]:


threats = train_toxic.loc[train_toxic['threat'] == 1]
insults = train_toxic.loc[train_toxic['insult'] == 1]
identity_hate = train_toxic.loc[train_toxic['identity_hate'] == 1]

print("threats: ", threats.comment_text)
print("\n\ninsults: ", insults.comment_text)
print("\n\nidentity_hate: ", identity_hate.comment_text)


# ## Natural Language Processing
# 
# The point of natural language processing is to clean up the text so only the relevant info is left that will allow the computer to better classify the data. For instance, irrelevant info such as the words 'a', 'the', 'your' etc. needs to be removed because it could effect the ultimate classification in a bad way. 
# 
# To do this I used the documentation from: https://pythonspot.com/nltk-stop-words/
# 
# In order to do natural language processing, I want to do the following steps:
# 
# 1. Get rid of all punctuation and irrelevant words (called stop words such as 'a', 'the', 'your' etc.)
# 2. (Possible step -- won't do it) Make any words into its root word (e.g. loved will become love)
# 3. Making everything lowercase so capitals don't have a large effect on how it is classified
# 
# Here I am going to show an example with one comment and will later apply it to every comment:

# In[12]:


first_comment = train_toxic['comment_text'][0]
first_comment


# In[13]:


comment = re.sub('[^a-zA-Z]', ' ', first_comment) # Taking out everything that is not a - z / A - Z and is not capital
comment = comment.lower() # making everything in the comment lowercase
comment = comment.split() # split the comment into different words so it becomes a list of the different words

# ps = PorterStemmer() (If I decided to do stemming I would add this)


# Taking out 'stop words' which are basically irrelevant words that will confuse the machine

# In[14]:


comment = [word for word in comment if not word in set(stopwords.words('english'))] 
comment = ' '.join(comment) # joining them all together

print("Original Comment: ", first_comment)
print("New comment: ", comment) 


# ## Cleaning up all the text 
# 
# Below I go through each comment and clean up all of the text to prepare it for training. 

# In[16]:


full_text = []

for i in range(len(train_toxic)):
    comment = re.sub('[^a-zA-Z]', ' ', train_toxic['comment_text'][i])
    comment = comment.lower() 
    comment = comment.split()
    
    comment = [word for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    
    full_text.append(comment)
    
train_toxic['comment_text'] = full_text


# The new cleaned up data:

# In[17]:


train_toxic


# ## Tokenization 
# 
# To go forward, there are two models I could prepare for. One is the LSTM model and the other is the 'Bag of Words' model. In order to prepare for both, I need to do the following now that my data is 'cleaned'. Simply, using a tokenizer allows me to divide my comments up into meaningful pieces. 
# 
# Here I start the process:

# In[18]:


targets = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
values = train_toxic[targets].values
training_setences = train_toxic["comment_text"]

tokenization = Tokenizer(num_words = 19500) # 19500 = num of unique words 
tokenization.fit_on_texts(list(training_setences)) 
training_after_tokenization = tokenization.texts_to_sequences(training_setences)


# ### Adjusting Sentence Length 
# 
# Now what I need to do is adjust the sentence length so each sentence is the same length so when I input it, it will work equally in the model and won't interefere with it. I can go about this in mutliple ways one of which is filling some sentences with 0's until it is the same length as others.
# 
# Therefore, I learned 'padding' which lets you make a max-length for some sentences (that are extremely long) and add 0's to sentences that are too short. I accomplish this through the following piece of code: 

# In[20]:


padding_training = pad_sequences(training_after_tokenization, 175)


# # The Model
# 
# Below I outline all the models I use. Much of the info that I learned to understand what RNN's/LSTM's were and there function in Machine learning was taken from this blog: http://colah.github.io/posts/2015-08-Understanding-LSTMs/ (however, NO code was taken from this blog and used). 
# 
# ## RNN + LSTM 
# 
# The first model I will use is a LSTM Neural Network which is based on the idea behind Reccuring Neural Networks and solves the vanishing gradient problem. 
# 
# **RNN's**: The whole concept of reccuring neural networks is to bascially remember what happened previously. So it is the same thing as a standard neural network but is able to remember what was in that neuron. This allows them to pass information onto themselves in the future and then remember things. For example, if you couldn't remember what happened in the previous presentation that would be pretty sad. That is your short term memory being enacted here and we want to use that same power with neural networks. Ultimately, it gives you previous contect. 
# 
# **The Vanishing Gradient Problem**: As we propogate the error through the network, it goes through many layers of neurons connected to themselves, it causes the gradient to decline rapidly meaning that the weights of the layers on the very far left are updated much slower than the weights on the far right creating a chain effect, effecting the whole thing. 
# 
# Basically, the vanishing gradient problem is where the RNN can remember things from the short term (and this is all that is necessary usually for a small amount of data) but as time goes on, it has trouble remembering things and slowly deteriates until it is eventually useless. 
# 
# **LSTM Solution**: In all practical terms, LSTM's are built specifically around long term memory. The way they accomplish this is very complex (see the blog cited above) and basically uses four neural networks or 'valves' in one to accomplish this. 
# 
# ![LSTM Solution Example](Desktop/LSTM Example.png)

# ## Layers I will Create:
# 
# The final classes I am using to create the ultimate neural network (all of these have been imported above):
# 
# 1. I will use the 'Dense' class to add the output layer
# 2. I will use the 'LSTM' class to add the LSTM layers (as discussed above)
# 3. I will use the 'Dropout' class to add dropout regularization 
# 
# Dropout Regularization basically allows me to reduce overfitting (only for neural networks) by dropping out units of the hidden layers and visible layers in the neural network. 
# 
# I also create standard input layers and embedding layers (a pretty confusing topic, here is a blog I found in addition to the LSTM one that is helpful: (https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/). 
# 
# Below I create it:  

# In[27]:


the_inputs = Input(shape=(175, )) # Input layer (175 = the maximum length for the size of a comment)
out = Embedding(19500, 125)(the_inputs) # Embedding layer -- to see the relevance and contect of any particular word (19500 is the max feautures)
out = LSTM(60, return_sequences=True, name='lstm_layer')(out) # Return sequences = True allows me to build more LSTM layers that can work together
out = GlobalMaxPool1D()(out) # To reshape the data from 3D to 2D


# The Dropout Layer: The dropout layer drops out units of the hidden layers and visible layers in the neural network. After research, I found that the average amount to dropout was 10-20%, and after some tweaking (seen at the end) to see what works best, I found that the optimal dropout layer was 12% for my dataset. 

# In[28]:


out = Dropout(0.12)(out)
out = Dense(52, activation="relu")(out) # Found 50 was the best after extensive grid search
out = Dropout(0.12)(out)
out = Dense(6, activation="sigmoid")(out) # Using sigmoid to get binary classification on each of the labels


# ## The LSTM Final Model:
# 
# Finally, to create the model, I used binary crossentropy because I needed to classify things as either '0's or '1's. 

# In[30]:


LSTM_model = Model(inputs = the_inputs, outputs = out)
LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Now while running it, after extensive grid search work, I found that the optimal batch size was 33 with the ideal number of epochs being 2. Validation split allowed me to split the training set into partly a test set so I can see how it performs as predictions. 

# In[31]:


LSTM_model.fit(padding_training, values, batch_size=33, epochs=2, validation_split=0.1)


# ## Grid Search 
# 
# Below I show an example of my grid search function. I ran it much, much more extensively for many of the parameters (including the Dropout amounts, batch size, epochs, max sentence length etc.) I change the function each time I run it, and have made some changes to where you will have to tweak it a bit for it to properly run. The untweaked function can be seen here below: 

# In[36]:


from sklearn.model_selection import GridSearchCV

def run_grid_search(model):
    grid = GridSearchCV(model,
                         param_grid={'epochs': [1, 2, 3, 4],
                                      'batch_size' : [31, 32, 33, 34]},
                         scoring='accuracy',
                         cv=3)

    grid.fit(the_inputs, out)
    grid_results = grid.cv_results_ 
    zipped_results = list(zip(grid_results["params"], grid_results["mean_test_score"])) 
    zipped_results_sorted = sorted(zipped_results, key=lambda tmp : tmp[1])[::-1]
    for element in zipped_results_sorted: 
        print(element[1], element[0])

    print('The parameters of the best model are: ') 
    print(grid.best_params_)


# # Conclusions
# 
# In the end, I learned a lot about understanding my data, natural language processing, using tokenization, using RNN models, the vanishing gradient problem and how LSTM solves that, and using grid search to effectively change the hyper parameter in my model. 
# 
# Ultimately, after extensive grid search work, I was able to get my model up to a 98% prediction accuracy on the *test set* I split it into! I learned a lot of new things and dipped my feet into complex models within deep learning...I hope to continue on this path. 
# 
# 

#  
