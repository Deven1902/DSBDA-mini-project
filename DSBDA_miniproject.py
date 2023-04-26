#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv')


# In[3]:


tfidf = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])


# In[4]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[5]:


def recommend_movies(title, cosine_sim=cosine_sim, df=df):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


# In[8]:


recommend_movies('Twilight')


# In[ ]:




