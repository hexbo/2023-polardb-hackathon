#Data manipulation
import pandas  as pd 
import numpy as np 

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

# Outlier Treatmment
from scipy.stats.mstats import winsorize
# Feature scaling
from sklearn.preprocessing import MinMaxScaler

# using gridserchcv to find best params
from sklearn.model_selection import ShuffleSplit,GridSearchCV,RandomizedSearchCV

# Train test split
from sklearn.model_selection import train_test_split

# Modelling
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import  RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Model Evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,roc_auc_score

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError

original_train_df = pd.read_csv('Data/train_BRCpofr.csv')

# creating copy to perform feature engineering in later stage
train_df = original_train_df.copy()

def check_variable_types(data):
    #target columns
    tar_col = ['cltv']

    #categorical features
    obj_cols = data.drop(tar_col,axis=1).select_dtypes(exclude=np.number).columns.to_list()
    cat_cols = []
    ord_cols = []

    for col in obj_cols:
        if data[col].value_counts().shape[0] == 2:
            cat_cols.append(col)
        elif data[col].value_counts().shape[0] > 2:
            ord_cols.append(col)

    #numerical features
    num_cols = data.drop(tar_col,axis=1).select_dtypes(include=np.number).columns.to_list()
    suspected_cat_cols = []

    #checking if any is categorial 
    for col in num_cols:
        if data[col].value_counts().shape[0] <= 5:
            print("Column name: ",col)
            print(data[col].value_counts())
            suspected_cat_cols.append(col)
            
    return tar_col,cat_cols,ord_cols,num_cols,suspected_cat_cols
    
tar_cols,cat_cols,ord_cols,num_cols,sus_cols = check_variable_types(train_df)

num_cols.remove('marital_status')
num_cols.remove('vintage')
cat_cols.extend(['marital_status','vintage'])

def cat_col_plot(data,cols):
    for col in cols:
        plt.figure(figsize=(8,4))
        data[col].value_counts(normalize=True).plot.bar(title=col)
        plt.tight_layout()
        plt.show()
     

cat_ord_cols = cat_cols + ord_cols

def num_cols_plot(data,cols):
    plt.figure(figsize=(10,3))
    for i in range(0,len(cols)):
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1)
        sns.distplot(data[cols[i]],kde=True)
        plt.subplot(1,3,2)
        sns.boxplot(data[cols[i]])
        plt.subplot(1,3,3)
        sns.violinplot(data[cols[i]])
        plt.tight_layout()
        plt.show()

def cat_cols_bivariate_plot(data,col,target):
    plt.figure(figsize=(10,4))
    plt.subplot(131)
    sns.barplot(x=col,y=target,data=data)
    plt.subplot(132)
    sns.violinplot(x=col, y=target, data=data,palette='rainbow')
    plt.subplot(133)
    sns.stripplot(x=col, y=target, data=data)
    plt.tight_layout()
    plt.show()

total_cat_cols = cat_cols + ord_cols

num_cols = train_df.select_dtypes(include=np.number).drop(columns=['marital_status', 'vintage'],axis=1).columns.to_list()

# #create a histogram
# for col in num_cols:
#     fig = px.histogram(train_df, x=col)
#     fig.show()

# Function to detect Outliers [Z score]
def detect_outliers_z_score(df):
    outliers = pd.DataFrame(columns=['Column','No. of Outlier'])
    cols = df.select_dtypes(include=np.number).columns.to_list()
    
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low_fence = df[c].mean() - 3*df[c].std()
        high_fence = df[c].mean() + 3*df[c].std()
        o = df[(df[c] < low_fence) | (df[c] > high_fence)].shape[0]

        outliers = outliers.append({'Column':c,'No. of Outlier':o},ignore_index=True)

    return outliers

Outliers_zscore = detect_outliers_z_score(train_df)
Outliers_zscore.set_index('Column',drop=True,inplace=True)
Outliers_zscore.sort_values(by='No. of Outlier',inplace=True)
Outliers_zscore.iloc[2:]

# Function to detect Outliers [IQR]
def detect_outliers_iqr(df):
    outliers = pd.DataFrame(columns=['Column','No. of Outlier'])
    cols = df.select_dtypes(include=np.number).columns.to_list()
    
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low_fence = q1 - (1.5*iqr)
        high_fence = q3 + (1.5*iqr)
        o = df[(df[c] < low_fence) | (df[c] > high_fence)].shape[0]

        outliers = outliers.append({'Column':c,'No. of Outlier':o},ignore_index=True)

    return outliers

Outliers_iqr = detect_outliers_iqr(train_df)
Outliers_iqr.set_index('Column',drop=True,inplace=True)
Outliers_iqr.sort_values(by='No. of Outlier',inplace=True)

def plot_violin(dataframe):
    numeric_cols = dataframe.select_dtypes(include=np.number).drop(columns=['marital_status', 'vintage'],axis=1).columns.to_list()
    dataframe = dataframe[numeric_cols]
    
    for i in range(0,len(numeric_cols),2):
        if len(numeric_cols) > i+1:
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            sns.violinplot(dataframe[numeric_cols[i]])
            plt.subplot(122)
            sns.violinplot(dataframe[numeric_cols[i+1]])
            plt.tight_layout()
            plt.show()
        else:
            sns.violinplot(dataframe[numeric_cols[i]])

df1 = train_df.copy()
#function to treat outliers

def treat_outliers(dataframe):
    col = ['claim_amount','cltv']
    for col in cols:
        dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))
    
    return dataframe    

df1 = treat_outliers(df1)
categorical_columns = train_df.select_dtypes('object').columns.to_list()

train_df_encode = pd.get_dummies(data = train_df, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
# Applying same label encoding on outlier treated df
categorical_columns = df1.select_dtypes('object').columns.to_list()
df1_encode = pd.get_dummies(data = df1, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')

mms = MinMaxScaler()
train_df_scaled = train_df_encode.copy()
train_df_scaled[['claim_amount']] = mms.fit_transform(train_df_scaled[['claim_amount']])
# Applying same scaling on outlier treated df
df1_scaled = df1_encode.copy()
df1_scaled[['claim_amount']] = mms.fit_transform(df1_scaled[['claim_amount']])

X = train_df_encode.drop(columns='cltv',axis=1) #Independent Variables
y = train_df_encode['cltv'] #Dependent Variable
# for outlier treated df
X_out = df1_scaled.drop(columns='cltv',axis=1) #Independent Variables
y_out = df1_scaled['cltv'] #Dependent Variable

# Creating a function for Model Building and selection
def find_best_model(X,y):
    models = {
        'linear_regression' : {
            'model' : LinearRegression()
        },
        'lasso' : {
            'model' : Lasso()
        },
        'decision_tree' : {
            'model' : DecisionTreeRegressor()
        },
        'random_forest' : {
            'model' : RandomForestRegressor()
        },
        'adaboost' : {
            'model' : AdaBoostRegressor()
        },
        'xgboost' : {
            'model' : XGBRegressor()
        }
    }
    
    scores = []
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    
    for model_name, model_params in models.items():
        reg_model = model_params['model'].fit(X_train,y_train)
        y_test_predict = reg_model.predict(X_test)
        y_train_predict = reg_model.predict(X_train)
        test_score = r2_score(y_test,y_test_predict)
        train_score = r2_score(y_train,y_train_predict)
        scores.append({
            'model': model_name,
            'Train_score' : train_score,
            'Test_score' : test_score
        })
        
    return pd.DataFrame(scores,columns=['model','Train_score','Test_score'])

hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.01
# Creating model using the Sequential in tensorflow
def build_model_using_sequential():
  model = Sequential([
    Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
    Dense(1, kernel_initializer='normal', activation='linear')
  ])
  return model
# build the model
model = build_model_using_sequential()

# loss function
msle = MeanSquaredLogarithmicError()
model.compile(
    loss=msle, 
    optimizer=Adam(learning_rate=learning_rate), 
    metrics=[msle]
)
# train the model
history = model.fit(
    X_out_scaled.values, 
    y_out.values, 
    epochs=50, 
    batch_size=64,
    validation_split=0.2
)

X = df1_scaled.drop(columns='cltv',axis=1) #Independent Variables
y = df1_scaled['cltv'] #Dependent Variable
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = XGBRegressor()
model.fit(X_train,y_train)

y_test_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)
test_score = r2_score(y_test,y_test_predict)
train_score = r2_score(y_train,y_train_predict)

def evaluation(ya,yp):
    mse = mean_squared_error(ya,yp)
    print("MSE: ",mse)
    print("*"*80)
    rmse = np.sqrt(mse)
    print("RMSE: ",rmse)
    print("*"*80)
    mae = mean_absolute_error(ya,yp)
    print("MAE: ",mae)
    print("*"*80)
    score = r2_score(ya,yp)
    print("R2 Score: ",score)

test_df = pd.read_csv('Data/test_koRSKBP.csv')
# Preprocessing test data
test_df_copy = test_df.copy()

test_df_copy.drop(columns=['id'],axis=1,inplace=True)

categorical_columns = test_df_copy.select_dtypes('object').columns.to_list()
test_df_encode = pd.get_dummies(data = test_df_copy, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')

test_df_encode.rename(columns={'OHE_High School':'OHE_High_School',
'OHE_5L-10L':'OHE_5L_to_10L',
'OHE_<=2L':'OHE_less_than_2L',
'OHE_More than 10L':'OHE_More_than_10L',
'OHE_More than 1':'OHE_More_than_1'},inplace=True)

test_df_encode[['claim_amount']] = mms.transform(test_df_encode[['claim_amount']])

# predicting
Test_Prediction = model.predict(test_df_encode)

#creating the sample submission file
df1 = test_df.copy()
df1['cltv']=np.around(Test_Prediction,2)
sample_submission = df1[['id','cltv']]
sample_submission.to_csv("sample_submission.csv",index=False)