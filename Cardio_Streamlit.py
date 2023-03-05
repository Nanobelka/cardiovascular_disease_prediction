#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[2]:


import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import joblib
# import pickle
# from PIL import Image


# ## Constants

# In[ ]:


PATH_DATA = ''
CR='\n'

# text styles
class f:
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    END = "\033[0m"


# ## Settings

# In[ ]:


st.set_page_config(
                   page_title='Cardiovascular disease prediction',
                   page_icon='⚕️',
                   initial_sidebar_state='expanded',
                   menu_items={
#                                'Get Help': 'https://....',
#                                'Report a bug': "https://....",
                               'About': 'Written by Sergei Vasiliev. Fell free contact to me in Telegram @nanobelkads.'
                              }
                 )


# ## Functions

# In[ ]:


@st.cache_resource
def load_model_local():

#     with open(f'{PATH_DATA}model_dump.pcl', 'rb') as model_dump:
#         model = pickle.load(model_dump)
        
    model = joblib.load(f'{PATH_DATA}model_dump.mdl')

    return model


# In[ ]:


@st.cache_data
def load_data_local(df_name, n_rows=5):
    df = pd.read_csv(f'{PATH_DATA}{df_name}', nrows=n_rows)
    return df


# In[ ]:


@st.cache_data
def target_encode(df_train, df_test, feature_list, target, agg_func_list=['mean'], fill_na=0.5):
    '''
    Принимает feature_list и делает для него target encoding,
    используя заданную агрегирующую функцию agg_func.
    '''
    
    for agg_func in agg_func_list:
    
        new_feature = '_'.join(feature_list) + '_TRG_' + agg_func

        df_train[new_feature] = df_train.groupby(feature_list)[target].transform(agg_func)
        df_train[new_feature] = df_train[new_feature].fillna(fill_na)

        df_test = df_test.merge(
                                df_train[feature_list + [new_feature]].drop_duplicates(),
                                on=feature_list,
                                how="left",
                               )
        df_test[new_feature] = df_test[new_feature].fillna(fill_na)

    return df_test


# ## Loads

# In[ ]:


# загрузка модели из файла
model = load_model_local()


# In[ ]:


# загрузка обучающих данных из файла
# приходится это делать, чтобы выполнить feature engineering
# можно заменить большой train на несколько небольших таблиц с агрегированными данными

data_train = load_data_local('EDA_train.csv', n_rows=None)


# In[ ]:


# # НЕ ИСПОЛЬЗУЕТСЯ (ПОКА)

# # загрузка предобработанных (после feature engineering) тестовых данных из файла
# data = load_data_local('FE_test.csv', n_rows=3)

# # прогноз для загруженных данных
# data['cardio'] = model.predict_proba(data)[:,1]
# # st.dataframe(data)


# ## Output basic info

# In[ ]:


# image = Image.open('banner.jpg')
# st.image(image)


# In[ ]:


st.image('banner.jpg')


# In[ ]:


# заголовок приложения
st.title('Cardiovascular disease prediction')

# пояснительный текст
st.text('Enter your details on the left folding panel.')
st.text('The prediction will change as data is entered.')

st.caption('Click on sign in the left upper corner if the panel is hidden.')

st.caption('------')


# ## Input data by user

# In[ ]:


# ввод данных с экрана

with st.sidebar:

    column_1, column_2 = st.columns(2)
    with column_1:
        gender_radio = st.radio('**Gender**', ['Male', 'Female'], key='gender_radio')
        gender = 0 if gender_radio == 'Male' else 1
    with column_2:
        age = st.selectbox('**Age**', range(40,66), key='age')
    
    column_1, column_2 = st.columns(2)
    with column_1:
        height = st.slider('**Height**', min_value=140, max_value=200, value=170, key='height')
    with column_2:
        weight = st.slider('**Weight**', min_value=40, max_value=150, value=70, key='weight')

    column_1, column_2 = st.columns(2)
    with column_1:
        ap_hi = st.slider('**Systolic blood pressure**', min_value=70, max_value=200, value=120, step=10, key='ap_hi')
    with column_2:
        ap_lo = st.slider('**Diastolic blood pressure**', min_value=40, max_value=ap_hi, step=10, key='ap_lo')

    column_1, column_2 = st.columns(2)
    with column_1:
        cholesterol_radio = st.radio('**Cholesterol**', ['normal','above normal','well above normal'], key='cholesterol')
        if cholesterol_radio == 'normal':
            cholesterol = 1 
        elif cholesterol_radio == 'above normal':
            cholesterol = 2
        else:
            cholesterol = 3
    with column_2:
        gluc_radio = st.radio('**Glucose**', ['normal','above normal','well above normal'], key='gluc')
        if gluc_radio == 'normal':
            gluc = 1 
        elif gluc_radio == 'above normal':
            gluc = 2
        else:
            gluc = 3

    habits = st.multiselect('**Bad and good habits**',
                            ['Smoking', 'Alcohol intake', 'Physical activity'],
                            ['Smoking', 'Alcohol intake', 'Physical activity']
                           )
    smoke = 1 if 'Smoking' in habits else 0
    alco = 1 if 'Alcohol intake' in habits else 0
    active = 1 if 'Physical activity' in habits else 0


# In[ ]:


# объединение введенных данных в мини-таблицу (из одной строки)

data_test = pd.DataFrame(data={'gender':[gender],
                               'age':[age],
                               'height':[height],
                               'weight':[weight],
                               'ap_hi':[ap_hi],
                               'ap_lo':[ap_lo],
                               'cholesterol':[cholesterol],
                               'gluc':[gluc],
                               'smoke':[smoke],
                               'alco':[alco],
                               'active':[active],
                              }
                        )


# ## Processing user's data

# In[ ]:


# feature engineering

for df in [data_train, data_test]:
    df['weight_bined'] = pd.cut(df.weight, bins=np.linspace(0, 200, 51), labels=False)
    df['height_bined'] = pd.cut(df.height, bins=np.linspace(0, 200, 51), labels=False)
    df['age_bined'] = pd.cut(df.age, bins=np.linspace(0, 100, 51), labels=False)
    df['aphi_bined'] = pd.cut(df.ap_hi, bins=np.linspace(0, 250, 26), labels=False)
    df['aplo_bined'] = pd.cut(df.ap_lo, bins=np.linspace(0, 150, 16), labels=False)


# In[ ]:


data_test = target_encode(data_train, data_test,
                                      feature_list=['gender','aphi_bined','aplo_bined'],
                                      target='cardio', agg_func_list=['mean'], fill_na=0.5)


# In[ ]:


data_test = target_encode(data_train, data_test,
                                      feature_list=['gender','age'],
                                      target='cardio', agg_func_list=['mean'], fill_na=0.5)


# In[ ]:


data_test = target_encode(data_train, data_test,
                                      feature_list=['gender','weight_bined','height_bined'],
                                      target='cardio', agg_func_list=['mean'], fill_na=0.5)


# In[ ]:


data_test = target_encode(data_train, data_test,
                                      feature_list=['gender','cholesterol','gluc'],
                                      target='cardio', agg_func_list=['mean'], fill_na=0.5)


# In[ ]:


data_test = target_encode(data_train, data_test,
                                      feature_list=['gender','active'],
                                      target='cardio', agg_func_list=['mean'], fill_na=0.5)


# In[ ]:


data_test = target_encode(data_train, data_test,
                                      feature_list=['gender','smoke'],
                                      target='cardio', agg_func_list=['mean'], fill_na=0.5)


# In[ ]:


data_test = target_encode(data_train, data_test,
                                      feature_list=['gender','alco'],
                                      target='cardio', agg_func_list=['mean'], fill_na=0.5)


# In[ ]:


# прогноз для введенных с экрана данных
data_test['cardio'] = model.predict_proba(data_test)[:,1]


# ## Output prediction

# In[8]:


# вывод результата

disease_proba = data_test.loc[0,"cardio"]

if disease_proba < 0.2:
    value_color = 'green'
elif disease_proba < 0.5:
    value_color = 'orange'
else:
    value_color = 'red'
    
st.subheader(f'Probability of cardiovascular disease is about :{value_color}[{disease_proba : .1%}]')


# In[ ]:


if disease_proba >= 0.5 and smoke == 1:
    st.write('**Бросай курить, вставай на лыжи!**')
    
if disease_proba >= 0.5 and alco == 1:
    st.write('**Надо меньше пить!**')


# ## Disclamer

# In[ ]:


st.caption('------')
st.caption('**Disclaimer.** The source of the data used for this application is unknown. Therefore, this application can under no circumstances be used for practical purposes. This application is made for demonstration purposes only.')


# ## Final service message

# In[ ]:


st.caption('------')
st.caption('*Service info: NO errors*')


# ## Remarks

# для удаленного запуска приложения из репозитория GitHub:

#     streamlit run https://raw.githubusercontent.com/Nanobelka/Cardiovascular-disease-prediction/main/Cardio_Streamlit.py

# для локального запуска приложения

#     d:  
#     cd DATA/Work_analytics/Jupyter_Notebook/Praktikum_DS/6_Cardio/Streamlit_app  
#     streamlit run Cardio_Streamlit.py  
