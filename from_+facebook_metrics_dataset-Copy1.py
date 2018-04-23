
# coding: utf-8

# In[ ]:

# In[567]:

import numpy as np
import pandas as pd
from zipfile import ZipFile
import urllib.request
from io import BytesIO


# In[568]:

url = urllib.request.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip")


# In[ ]:

with ZipFile(BytesIO(url.read())) as archive:
    print(archive.namelist())


# In[569]:

with ZipFile(BytesIO(url.read())) as archive:
    dataframe = pd.read_csv(archive.open('dataset_Facebook.csv'), sep = ';')


# In[ ]:

Посчитаем необходимые значения для всех 3-х типов записей по каждому характеризующему столбцу. Для указанных характеристик 
хватит штатных методов pandas
Для 'Type', 'Paid' и 'Category' не имеет смысла считать все, кроме 'mode', т.к. переменная принимает известные значения


# In[ ]:

Т.к. у встроенного метода mode() есть ряд недостатков, компромиссный вариант - функция, возвращающая наиболее часто
встречающееся значение (значения, если есть несколько с одинаковой частотой) и, для оценки наглядности - его частоту. Допущение: мода засчитывается отсутствующей, если повторяющихся значений не было или частота была меньше sqrt(кол-во наблюдений) - чтобы исключить явные случайные совпадения


# In[570]:

dataframe[1:5]


# In[571]:

def get_mode(series, by_type):
    v = series.value_counts()
    if by_type == False:
        max_freq = max(v.values)
        m = list(v.loc[v.values == max_freq].index)
        return (max_freq, m) if max_freq>=20 else 'NaN'
    else:
        modes = []
        for typ in v.index.levels[0]:
            max_freq = max(v[typ].values)
            m = list(v[typ].loc[v[typ].values == max_freq].index)
            if max_freq>=5:
                modes.append({max_freq: m})
            else:
                modes.append('NaN')
        return modes


# In[572]:

def get_stats(stats, series, func, by_type):
    shortcuts = {'mean': series.mean(), 'max': series.max(), 'min': series.min(), 'median': series.median(),
                 'mode': func(series, by_type)}
    for stat in shortcuts:
        stats[stat] = shortcuts[stat]
    return stats


# In[573]:

outliers1 = ['Type', 'Paid', 'Category']
metrics = list(dataframe.columns.values)
index0 = ['mean', 'max', 'min', 'median', 'mode']
stats = pd.DataFrame(index = index0, columns = metrics)
for metric in metrics:
    if metric in outliers:
        stats.loc['mode'][metric] = get_mode(dataframe[metric], by_type = False)
    else:
        stats.loc[ :, metric] = get_stats( stats.loc[ :, metric], dataframe[metric], get_mode, False)


# In[574]:

stats


# In[ ]:

Те же статистики отдельно для каждого типа переменной "Type". В данном случае моду для 'Category' также можно не считать:


# In[575]:

seq = [(stat, type) for stat in index1 for type in list(dataframe['Type'].unique())]
multiindex = pd.MultiIndex.from_tuples(seq, names = ['Stat', 'Type'])
specs_stats = pd.DataFrame(index = multiindex, columns = metrics)
specs_dataframe = dataframe.groupby('Type')


# In[576]:

for metric in metrics:
    if metric == 'Type':
        None
    elif metric == 'Category' or 'Paid':
        specs_stats.loc[:, metric]['mode'] = get_mode(specs_dataframe[metric], by_type = True)
    else:
        specs_stats.loc[ :, metric] = get_stats( specs_stats.loc[ :, metric], specs_dataframe[metric], get_mode, True)


# In[577]:

specs_stats


# In[578]:

dataframe


# In[ ]:

Из данного массива можно получить много информации. Самый популярный объект в выборке является публикацией на Фейсбуке. Тк Фейсбук использует сложный алгоритм сортировки постов, время публикации поста можно не учитывать при оценке его популярности. 
Нас интересует популярность поста у всех пользователей, а не только у пользователей, подписанных на страницу. Пост появляется в новостной ленте у новых пользователей если он: 1) набирает много лайков, репостов или комментариев 2) лайкнул/прокомментировал кто-то из друзей 3) это оплаченная реклама (видимо, метрика 'Paid'). Т.е., если, при прочих допускаемо равных, у одного поста в метрике 'Paid' стоит '0', а у второго '1', первый был популярнее. 


# In[ ]:

Таким образом, отправной характеристикой популярности поста можно считать сумму метрик 'like'+'share'+'comment'. Затем, можно
сверить метрику 'Paid'. Получим:


# In[579]:

dataframe['Popularity'] = dataframe['like'] + dataframe['share'] + dataframe['comment']


# In[580]:

dataframe[['Lifetime Post Total Reach', 'Paid', 'Popularity']].sort_values('Popularity', ascending = False)[0:5]


# In[ ]:

Как видим, наиболее популярным постом был №244, он же имел самое высокое кол-во просмотров пользователями (по предположению,
уникальными). Первый следующий за ним неоплаченный пост №168 имел в 3,2 раза меньше просмотров и в 4,3 раза меньше показов. Можно найти линейный фактор, очень грубо характеризующий увеличение вероятности поста быть показанным, если он был оплачен, либо для этой же цели прогнать простую регрессию (я посчитала это лишним), чтобы посмотреть не был ли пост №244 явным выбросом и, следовательно, на самом деле более популярным, чем пост №244:


# In[581]:

factor = dataframe[dataframe['Paid']==1][['Popularity',
                    'Lifetime Post Total Reach']].mean() / dataframe[dataframe['Paid']==0][['Popularity',
                    'Lifetime Post Total Reach']].mean()
factor


# In[ ]:

Как видим, средняя популярность оплаченных постов выше в полтора раза, чем у неоплаченных. Следовательно, вероятностью того, что
пост №168 мог быть самым популярным, можно предварительно пренебречь.

