# Модель и [приложение](https://github.com/Nanobelka/cardiovascular_disease_prediction) для прогнозирования вероятности сердечно-сосудистых заболеваний  

## На основании параметров обследованных пациентов спрогнозировать вероятность наличия сердечно-сосудистого заболевания.

### [Exploratory data analysis](https://github.com/Nanobelka/Yandex_Praktikum/blob/main/cardio/Cardio_1_EDA.ipynb)  
### [Feature engineering](https://github.com/Nanobelka/Yandex_Praktikum/blob/main/cardio/Cardio_2_FE.ipynb)  
### [Model selection and Final model](https://github.com/Nanobelka/Yandex_Praktikum/blob/main/cardio/Cardio_3A_Model_Selection_XGBC.ipynb)  
### [Application](https://cardiovascular-disease-prediction.streamlit.app/)

**Входные данные:** параметры обследованных пациентов; данные изначально разделены на обучающий и тестовый наборы.

**Цель проекта:**  
Predict risk of heart diseases from patient lifestyle information.  
Целевая метрика – ROC-AUC-score.

------

**Задачи проекта:** 

[Exploratory data analysis](https://github.com/Nanobelka/Yandex_Praktikum/blob/main/cardio/Cardio_1_EDA.ipynb)  
- провести исследовательский анализ данных;  
- исправить ошибки в данных;  
- подготовить данные для feature engineering  

[Feature engineering](https://github.com/Nanobelka/Yandex_Praktikum/blob/main/cardio/Cardio_2_FE.ipynb)  
- исследовать варианты создания новых признаков;  
- оценить значимость исходных и новых признаков для модели;  
- подготовить данные для обучения модели.  

[Model selection and Final model](https://github.com/Nanobelka/Yandex_Praktikum/blob/main/cardio/Cardio_4_Model_Selection_XGBC.ipynb)  
- сделать несколько вариантов моделей и выполнить сравнительный запуск;  
- оценить приемлемое время обучения моделей;  
- оценить дополнительные метрики моделей;  
- сохранить параметры обученных моделей;  
- выбрать модель для углубленного изучения;  
- оценить влияние признаков, при необходимости вернуться к этапу Feature Engineering;  
- сохранить обученную модель;  
- подготовить и сохранить submit file.

[Application](https://github.com/Nanobelka/cardiovascular_disease_prediction/blob/main/Cardio_Streamlit.py)
- используя обученную модель, создать [Streamlit-приложение]((https://cardiovascular-disease-prediction.streamlit.app/)) ([репозиторий приложения](https://github.com/Nanobelka/cardiovascular_disease_prediction))
