import pickle

import warnings

import numpy as np

import pandas as pd

import xgboost as xgb

import joblib

import streamlit as st

import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    FunctionTransformer,
    
)
sklearn.set_config(transform_output = 'pandas')

from feature_engine.outliers import Winsorizer
from feature_engine.encoding import (
    RareLabelEncoder,
    MeanEncoder,
    CountFrequencyEncoder
)
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance

import matplotlib.pyplot as plt


# Preprocessing operations

# airline

air_tranformer = Pipeline(steps=[
    # in this step we handle the all missing values with most frequent categorey
    ('imputer',SimpleImputer(strategy='most_frequent')),
    # in this step we group the rare categories with name other
    ('grouper',RareLabelEncoder(tol=0.1,replace_with='other',n_categories=2)),
    # in this step we apply the onehot encoder to convert the categories into numbers
    ('encoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])
# air_tranformer.fit_transform(x_train.loc[:,['airline']])

# date_of_journy

feature_to_extract = ['month','week','day_of_week','day_of_year']

doj_transformer = Pipeline(steps=[
    ('dt',DatetimeFeatures(features_to_extract=feature_to_extract,yearfirst=True,format='mixed')),
    ('scaler',MinMaxScaler())
])
# doj_transformer.fit_transform(x_train.loc[:,['date_of_journey']])

# source & destination

location_pipe = Pipeline(steps=[
    # we label the rare values with name other
    ('grouper',RareLabelEncoder(tol=0.1,replace_with='other',n_categories = 2)),
    # Now we perform mean encoding it calculate the mean value of each category regarding target column
    ('encoder',MeanEncoder()),
    ('transformer',PowerTransformer())
])
# location_pipe.fit_transform(location_subset,y_train)

def is_north(data):
    columns = data.columns.to_list()
    north_cities = ['Delhi','Kolkata','Mumbai','New Delhi']
    return(
        data
        .assign(**{
            f'{col}_is_north' : data.loc[:,col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns = columns)
    )

# is_north(location_subset)
# FunctionTransformer(func=is_north).fit_transform(location_subset)

location_transformer = FeatureUnion(transformer_list=[
    ('part1',location_pipe),
    ('part2',FunctionTransformer(func=is_north))
])
# location_transformer.fit_transform(location_subset,y_train)

# dept_time & arrival_time

time_pipe = Pipeline(steps=[
    ('dt',DatetimeFeatures(features_to_extract=['hour','minute'])),
    ('scaler',MinMaxScaler())
])
# time_pipe.fit_transform(time_subset)

def part_of_day(data,morning=4,noon=12,evening=16,night=20):
    columns = data.columns.to_list()
    x_temp = data.assign(**{
        col :pd.to_datetime(data.loc[:,col]).dt.hour
        for col in columns
    })
    return (
        x_temp
        .assign(**{
            f'{col}_part_of_day' :np.select(
                [x_temp.loc[:,col].between(morning,noon,inclusive='left'),
                x_temp.loc[:,col].between(noon,evening,inclusive='left'),
                x_temp.loc[:,col].between(evening,night,inclusive='left')],
                ['morning','afternoon','evening'],
                default='night'
                
            )
            for col in columns
        })
        .drop(columns=columns)
    )
# part_of_day(time_subset)
# FunctionTransformer(func=part_of_day).fit_transform(time_subset)

time_pipe2 = Pipeline(steps=[
    ('part',FunctionTransformer(func=part_of_day)),
    ('encoder',CountFrequencyEncoder()),
    ('scaler',MinMaxScaler())
])
# time_pipe2.fit_transform(time_subset)

time_transformer = FeatureUnion(transformer_list=[
    ('part1',time_pipe),
    ('part2',time_pipe2)
])
# time_transformer.fit_transform(time_subset)

# duration

class RBFPercentileSimilarity(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None,percentile=[.25,.50,.75],gamma=0.1):
        self.variables = variables
        self.percentile = percentile
        self.gamma = gamma


    def fit(self,X,y=None):
        if not self.variables:
            self.variables = X.select_dtypes(include='number').columns.to_list()

        self.refrence_values_ = {
            col:X.loc[:,col].quantile(self.percentile).values.reshape(-1,1)
            for col in self.variables
        }
        return self


    def transform(self,X):
        objects = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(percentile*100)}" for percentile in self.percentile ]
            obj = pd.DataFrame(
                data = rbf_kernel(X.loc[:,[col]],Y=self.refrence_values_[col],gamma = self.gamma),
                columns = columns
            )
            objects.append(obj)
        return pd.concat(objects,axis=1)
    
    
def duration_catgory(data,short=180,med=400):
    return (
        data
        .assign(
            duration_cat=np.select([data.duration.lt(short),
                                    data.duration.between(short,med,inclusive='left')],
                                    ['short','medium'],
                                    default='long')
        )
        .drop(columns='duration')
    )


duration_pipe1 = Pipeline(steps=[
    ('rbf',RBFPercentileSimilarity()),
    ('scaler',PowerTransformer())
])

duration_pipe2 = Pipeline(steps=[
    ('cat',FunctionTransformer(func=duration_catgory)),
    ('encoder',OrdinalEncoder(categories=[['short','medium','long']]))
])
duration_union = FeatureUnion(transformer_list=[
    ('part1',duration_pipe1),
    ('part2',duration_pipe2),
    ('part3',StandardScaler()),
])

duration_transformer = Pipeline(steps=[
    ('outliers',Winsorizer(capping_method='iqr',fold=1.5)),
    ('imputer',SimpleImputer(strategy='median')),
    ('union',duration_union)
])
# duration_transformer.fit_transform(x_train.loc[:,['duration']])

# total_stops

def is_direct(X):
    return(
        X
        .assign(
            is_direct_flight = X.total_stops.eq(0).astype(int)
        )
    )

total_stop_transformer = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('fun',FunctionTransformer(func=is_direct))
])
# total_stop_transformer.fit_transform(x_train.loc[:,['total_stops']])

# additiona_info

info_pipe1 = Pipeline(steps=[
    ('group',RareLabelEncoder(tol=0.1,n_categories=2,replace_with='other')),
    ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
])
# info_pipe1.fit_transform(x_train.loc[:,['additional_info']])

def have_info(data):
    return (
        data
        .assign(additional_info = data.additional_info.ne('No Info').astype(int))
    )

info_union = FeatureUnion(transformer_list=[
    ('part1',info_pipe1),
    ('part2',FunctionTransformer(func=have_info))
])


info_transformer = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='constant',fill_value='unknown')),
    ('union',info_union)
])
# info_transformer.fit_transform(x_train.loc[:,['additional_info']])

# colummn transformer

column_transformer = ColumnTransformer(transformers=[
    ('air',air_tranformer,['airline']),
    ('doj',doj_transformer,['date_of_journey']),
    ('location',location_transformer,['source','destination']),
    ('time',time_transformer,['dep_time','arrival_time']),
    ('duration',duration_transformer,['duration']),
    ('stops',total_stop_transformer,['total_stops']),
    ('info',info_transformer,['additional_info'])
],remainder='passthrough')

# feature selector

estimator = RandomForestRegressor(n_estimators=10,max_depth=3,random_state=42)

selector = SelectBySingleFeaturePerformance(
    estimator = estimator,
    scoring = 'r2',
    threshold = 0.1
)

# preprocessor

preprocessor = Pipeline(steps=[
    ('ct',column_transformer),
    ('select',selector)
])


# preprocessor.fit(
#     train.drop(columns = 'price'),
#     train.price.copy()
# )

#read the training data

train = pd.read_csv('train_data.csv')
X_train = train.drop(columns = 'price')
y_train = train.price.copy()

# fit and save the preprocessor

preprocessor.fit(X_train,y_train)

joblib.dump(preprocessor,'preprocessor.joblib')


# create web application
# we use set_page_config for page configuration 
st.set_page_config(
    page_title='Flight Prices Prediction',
    page_icon='✈️',
    layout='wide'
)

# Now we st the title of our app
st.title('Flight Prices Prediction')
# Now we take the input columns a sinput
# user inputs
# we use st.select for categorical values
airline = st.selectbox(
    'Airline:',
    options=X_train.airline.unique()

)

# date of journey
doj = st.date_input('Date of Journey:')

# soure and destination
source = st.selectbox(
    'Sourec:',
    options=X_train.source.unique()
)

destination = st.selectbox(
    'Destination:',
    options=X_train.destination.unique()
)

# Depature time and arrival time
dep_time = st.time_input('Depature Time:')
arrival_time = st.time_input('Arrival Time:')

# duration
duration = st.number_input(
    'Duratin (min):',
    step =1
)

# total stops
total_stops = st.number_input(
    'Total Stops:',
    step =1,
    min_value=0
)

# Additional Info
additional_info = st.selectbox(
    'Additional Info:',
    options=X_train.additional_info.unique()
)

# Now we convert our data to pandas dataframe to transform the data
# This is our data frame based on the user input
# pandas date time funtion will not work on streamlit date time object
# so we convert our data date time columns convert into string 
# because it work with stream lit
x_new = pd.DataFrame(dict(
    airline = [airline],
    date_of_journey = [doj],
    source = [source],
    destination = [destination],
    dep_time = [dep_time],
    arrival_time = [arrival_time],
    duration = [duration],
    total_stops = [total_stops],
    additional_info = [additional_info]
    # Now we convert datetime columns into string 
    # to make these columns streamlit compatible
    # This code convert these three column into string
)).astype({
    col:'str'
    for col in ['date_of_journey','dep_time','arrival_time'] 
})

# Now we preprocees the x_new data frame who we get from user
# Our preprocesser are save in preprocessor.joblib file so load this file to 
# preprocess the user data
# Now we load joblibfile for preprocessing
# when user click on predict butto we do all the following things
if st.button('Predict'):

    saved_preprocessor = joblib.load('preprocessor.joblib')
    # it will perform the transformation on user input
    x_new_pre = saved_preprocessor.transform(x_new)

    # Now we get the prediction
    # For finding prediction we need the model so we load the model
    # rb means we read our model in binary format

    with open('xgboost-model','rb') as f:
        model = pickle.load(f)
        # our model predict on x_new_pre but we canot use it directly
        # so we use xgb.DMatrix to transform first then use our data
    x_new_xgb = xgb.DMatrix(x_new_pre,enable_categorical=True)
    pred = model.predict(x_new_xgb)[0]
# info is function that show all the things in blue box
# This code showing the result
    st.info(f'The Predicted Price is {pred:,.0f} PKR')