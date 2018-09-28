from flask import render_template
from flaskflickr import app
import pandas as pd
from flask import request

from flaskflickr.a_Model import dest2vec_recommendation

import sys
sys.path.append('/home/xavier/DIR.insight2018/')
from secret import API_KEY, API_SECRET
import flickrapi
import time
import datetime
from calendar import monthrange
import pickle
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from country_list import countries_for_language
import unidecode # to remove accent on letters
from collections import Counter  # For the histogram
import matplotlib.pyplot as plt # For the histogram
import datapackage # for the list of cities in the world
import re
# import scipy.sparse as sps
import subprocess
from surprise import Reader, Dataset
import gensim   

def lower_decode(b):
    lower_case = [x.lower() for x in b]
    decoded =  [unidecode.unidecode(x) for x in lower_case]
    return decoded
def country_names():
#    languages = ["en","fr","de","es"]
    languages = ["en"]
    b=[]
    for lang in languages:
        countries = dict(countries_for_language(lang))
        tmp=list(countries.values())
        b+=tmp
    decoded = lower_decode(b)
    return set(decoded)
countries = country_names()

def city_names():
    data_url = 'https://datahub.io/core/world-cities/datapackage.json'

    # to load Data Package into storage
    package = datapackage.Package(data_url)

    # to load only tabular data
    resources = package.resources
    for resource in resources:
        if resource.tabular:
            data = pd.read_csv(resource.descriptor['path'])
#            print (data)
    b=data['name']
    decoded = lower_decode(b)
    return set(decoded)
cities = city_names()

# Remove a few city names because they are confusing: holiday, of, green, panorama
for w in ["holiday", "of", "green", "panorama",'opportunity', 'sunrise', 'paradise', 'man','sunset']:
    cities.remove(w)

with open('seq_travel.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    seq_travel = pickle.load(f)

trip_sequence = list(seq_travel.values())
model = gensim.models.Word2Vec(trip_sequence)
model.train(trip_sequence, total_examples=len(trip_sequence), epochs=10)

def dest2vec_recommendation(dest_1):
    tmp = model.most_similar(dest_1)[0:3]
    output = [item[0].capitalize() for item in tmp]
    return(output)


# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
# user = 'postgres' #add your Postgres username here
# password = '' 
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user, host = host, password = password)

@app.route('/')
@app.route('/index')
def index():
   return render_template("index.html",
      title = 'Home', user = { 'nickname': 'Miguel' },
      )

@app.route('/input')
#def cesareans_input():
def flickr_project_input():
   return render_template("input.html")

@app.route('/output')
#def cesareans_output():
def flickr_project_output():
   list_dest=[]
 #pull 'birth_month' from input field and store it
   for idx in range(1,6):
      dest = request.args.get(('dest_'+str(idx)))
      dest_lower = lower_decode([dest])[0]
      if (dest_lower in cities) | (dest_lower in countries):
         list_dest.append(dest_lower)
   sugg = dest2vec_recommendation(list_dest)
   linked_sugg="<br>"
   for i in range(len(sugg)):
      linked_sugg += '<a href=' + '"https://wikitravel.org/en/' +  \
              sugg[i] +'">' + sugg[i] +'</a> <br>'
   if len(list_dest) > 1:
      dest_pretty = [item.capitalize() for item in list_dest]
   else:
      dest_pretty = [item[0].capitalize() for item in [list_dest]] # Works for one city
   #just select the Cesareans  from the birth dtabase for the month that the user inputs
# query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
# print(query)
# query_results=pd.read_sql_query(query,con)
# print(query_results)
# births = []
# for i in range(0,query_results.shape[0]):
#     births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#     the_result = ModelIt(patient,births)
# return render_template("output.html", births = births, the_result = the_result)
#   return render_template("output.html",dest_1 = ", ".join(dest_pretty), sugg = ", ".join(sugg))
   return render_template("output.html",dest_1 = ", ".join(dest_pretty), sugg = linked_sugg )

