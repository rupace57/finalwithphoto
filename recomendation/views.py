import os
from django import http
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

file='FinalData.csv'
file2='rating.csv'

def index (request):
    return render(request,'index.html')
def books (request):
    # return HttpResponse("You search")
    try:
        user_search=request.GET.get("u_search")
        df=pd.read_csv(file)
        result=df.loc[df["title"]==user_search, :]
        a=result.iloc[0][3]
        rec=df.loc[df["Genres"]==a,["book_id","title","authors"]]
        reindex=rec.reset_index(drop=True)
        return HttpResponse( reindex.to_html())
    except:
        user_search=request.GET.get("u_search")
        df=pd.read_csv(file)
        result=df.loc[df["title"].str.contains(user_search) , :]
        reindex=result.reset_index(drop=True)
        return HttpResponse(reindex.to_html())

def Biography(request):
    df=pd.read_csv(file)
    rec=df.loc[df["Genres"]=="Biography",["book_id","title","authors"]]
    reindex=rec.reset_index(drop=True)
    return HttpResponse( reindex.to_html())
def Drama(request):
    df=pd.read_csv(file)
    rec=df.loc[df["Genres"]=="Drama",["book_id","title","authors"]]
    reindex=rec.reset_index(drop=True)
    return HttpResponse( reindex.to_html())
def Fantasy(request):
    df=pd.read_csv(file)
    rec=df.loc[df["Genres"]=="Fantasy",["book_id","title","authors"]]
    reindex=rec.reset_index(drop=True)
    return HttpResponse( reindex.to_html())
def Fiction(request):
    df=pd.read_csv(file)
    rec=df.loc[df["Genres"]=="Fiction",["book_id","title","authors"]]
    reindex=rec.reset_index(drop=True)
    return HttpResponse( reindex.to_html())
def Romance(request):
    df=pd.read_csv(file)
    rec=df.loc[df["Genres"]=="Romance",["book_id","title","authors"]]
    reindex=rec.reset_index(drop=True)
    return HttpResponse( reindex.to_html())
def Scifi(request):
    df=pd.read_csv(file)
    rec=df.loc[df["Genres"]=="SciFi",["book_id","title","authors"]]
    reindex=rec.reset_index(drop=True)
    return HttpResponse( reindex.to_html())
def Thriller(request):
    df=pd.read_csv(file)
    rec=df.loc[df["Genres"]=="Thriller",["book_id","title","authors"]]
    reindex=rec.reset_index(drop=True)
    return HttpResponse( reindex.to_html())
def cossim(request):
    user_search=request.GET.get("search")
    df_title=pd.read_csv(file)
    col_list= ["book_id", "title"]
    df_title['corpus'] = (pd.Series(df_title[['authors', 'Genres']].fillna('').values.tolist()).str.join(' '))
    tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix_corpus = tf_corpus.fit_transform(df_title['corpus'])
    cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

    titles = df_title['title']
    indices = pd.Series(df_title.index, index=df_title['title'])

    idx = indices[user_search]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    haah = pd.DataFrame(sim_scores, columns = ['index','cosim score'])
    book_indices = [i[0] for i in sim_scores]
    a=titles.iloc[book_indices][:21]
    a.to_frame()
    b=pd.merge(a,haah,left_index=True,right_on='index')
    del b['index']
    rec = b
    # rec2=pd.Series.to_frame(rec)
    # reindex=rec2.reset_index(drop=True)
    return HttpResponse( rec.to_html())
def svd(request):
    user_search=request.GET.get("search")
    df = pd.read_csv(file2, encoding = "ISO-8859-1")
    df_title =  pd.read_csv(file, encoding = "ISO-8859-1")
    f = ['count','mean']

    df_movie_summary = pd.DataFrame(df.groupby('book_id')['rating'].agg(f))

    df_cust_summary = df.groupby('user_id')['rating'].agg(f)



    df_p = pd.pivot_table(df,values='rating',index='user_id',columns='book_id')

    i = int(df_title.index[df_title['title'] == user_search][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df_title).join(df_movie_summary)[['title', 'PearsonR','count']]
    rec=corr_target[corr_target['count']>=0][1:21]
    del rec['count']

    a=pd.DataFrame(rec)
    reindex=a.reset_index(drop=True)
    return HttpResponse(reindex.to_html())
