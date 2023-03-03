#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import streamlit as st
# import pickle
import joblib
import sklearn
import xgboost

# In[ ]:


# PATH_APP_LOCAL = 'D:/DATA/Work_analytics/Jupyter_Notebook/Praktikum_DS/6_Cardio/Streamlit_app/'
# PATH_DATA_LOCAL = 'D:/DATA/Work_analytics/Jupyter_Notebook/Praktikum_DS/6_Cardio/datasets/'

PATH_APP_LOCAL = ''
PATH_DATA_LOCAL = ''

CR='\n'


# In[ ]:


@st.cache_resource
def load_model_local():

#     with open(f'{PATH_DATA_LOCAL}model_dump.pcl', 'rb') as model_dump:
#         model = pickle.load(model_dump)
        
   model = joblib.load(f'{PATH_DATA_LOCAL}model_dump.mdl')

    return model
    


# In[ ]:


@st.cache_data
def load_data_local(df_name, n_rows=5):
    df = pd.read_csv(f'{PATH_DATA_LOCAL}{df_name}', nrows=n_rows)
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


# In[ ]:





# In[ ]:


# заголовок приложения
st.title('Cardiovascular disease prediction')

# пояснительный текст
st.text(f'Enter your details on the left side of the screen.{CR}The prediction will change as data is entered.')


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


# In[ ]:





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
        cholesterol = st.radio('**Cholesterol**', [1,2,3], key='cholesterol')
    with column_2:
        gluc = st.radio('**Glucose**', [1,2,3], key='gluc')

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





# In[ ]:





# In[ ]:





# In[ ]:


'NO errors'


# для удаленного запуска приложения из репозитория GitHub:

#     streamlit run https://github.com/Nanobelka/Cardiovascular-disease-prediction/blob/main/Cardio_Streamlit.py

#     streamlit run https://raw.githubusercontent.com/Nanobelka/Cardiovascular-disease-prediction/main/Cardio_Streamlit.py

# для локального запуска приложения

#     d:  
#     cd DATA/Work_analytics/Jupyter_Notebook/Praktikum_DS/6_Cardio/Streamlit_app  
#     streamlit run Cardio_Streamlit.py  
