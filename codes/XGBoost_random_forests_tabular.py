#!/usr/bin/env python
# coding: utf-8
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pickle
from xgboost import XGBClassifier
import argparse
import itertools
import warnings
warnings.filterwarnings('ignore')
# "error", "ignore", "always", "default", "module" or "once"

def add_args(parser):

    
    parser.add_argument('--patient_file_name', type=str, default='/home/dsv/maul8511/Desktop/paper-8/final/all_data.csv',
                        help='tabular data file')
    
    parser.add_argument('--save_dir', type=str, default='',
                        help='result directory')
    
    parser.add_argument('--tabular_data_type', type=str,
                        help='tabular data type')
    
    parser.add_argument('--is_training', type=int, default=1,
                        help='training or testing')
                        
    parser.add_argument('--is_header', type=int, default=0,
                        help='writing the header line')

    parser.add_argument('--evaluation_metric', type=str, default='roc_auc',
                        help='evaluation metric, roc_auc, average_precision')
    
    args = parser.parse_args()
    
    return args


def get_clean_splitted_tabular_data(data_columns_dict, choice, data_file_name, no_stratification=False, id_header='ID', label_header='PCR Result', positive_text='Positive'):
    
    df = pd.read_csv(data_file_name)

    df[['Blood Pressure S', 'Blood Pressure D']] = df['Blood Pressure'].str.split('/', 1, expand=True).astype(float)
    
    df['SpO2']=df['SpO2'].str.strip('%').astype(float)
    
    df = df.replace('NA', 0)
    
    df = df.replace('none', 0)
    df = df.replace(np.nan, 0)
    
    cat_names = data_columns_dict[choice]['categorical_column_names']
    
    selected_column_names = data_columns_dict[choice]['categorical_column_names'] + data_columns_dict[choice]['numeric_column_names']

    df[cat_names] = df[cat_names].astype('category')

    df[cat_names] = df[cat_names].apply(lambda x: pd.factorize(x)[0])

    df[selected_column_names] = df[selected_column_names].astype(float)

    df = df[selected_column_names+[label_header]+[id_header]]
                
    df[label_header] = df[label_header].apply(lambda x: 1 if x == positive_text else 0)
    
    df.fillna(0)
    
    df_id_class_pos = df[df[label_header] == 1]
    df_id_class_neg = df[df[label_header] == 0]

    #Creating a dataframe with 10% values of original dataframe, test data
    df_id_class_pos_test = df_id_class_pos.sample(frac = 0.1, random_state=1969)
    df_id_class_neg_test = df_id_class_neg.sample(frac = 0.1, random_state=1969)
    
    df_test = pd.concat([df_id_class_pos_test, df_id_class_neg_test])

    #Creating dataframe with rest of the 90% values, train+valid data
    df_id_class_pos_train_valid = df_id_class_pos.drop(df_id_class_pos_test.index)
    df_id_class_neg_train_valid = df_id_class_neg.drop(df_id_class_neg_test.index)
    
        
    #Creating a dataframe with 10% values of train+valid dataframe, valid data
    df_id_class_pos_valid = df_id_class_pos_train_valid.sample(frac = 0.1, random_state=1969)
    df_id_class_neg_valid = df_id_class_neg_train_valid.sample(frac = 0.1, random_state=1969)
    
    df_valid = pd.concat([df_id_class_pos_valid, df_id_class_neg_valid])
    
    #Creating dataframe with rest of the 80% values, train data
    df_id_class_pos_train = df_id_class_pos_train_valid.drop(df_id_class_pos_valid.index)
    df_id_class_neg_train = df_id_class_neg_train_valid.drop(df_id_class_neg_valid.index)
    
    df_train = pd.concat([df_id_class_pos_train, df_id_class_neg_train])
    
    scaler = StandardScaler()
    
    y_train=df_train[label_header].to_numpy()
    
    df_train= df_train[selected_column_names]
    
    normalized_df_train = pd.DataFrame(
    scaler.fit_transform(df_train),
    columns = df_train.columns
    )
    
    y_test=df_test[label_header].to_numpy()
    
    df_test= df_test[selected_column_names]
    
    normalized_df_test = pd.DataFrame(
    scaler.transform(df_test),
    columns = df_test.columns
    )
    
    y_valid=df_valid[label_header].to_numpy()
    
    df_valid= df_valid[selected_column_names]
    
    normalized_df_valid = pd.DataFrame(
    scaler.transform(df_valid),
    columns = df_valid.columns
    )
    
    tabular_data = dict()
                
    tabular_data['y_train']=y_train
    tabular_data['X_train']=normalized_df_train.to_numpy()
    tabular_data['y_test']=y_test
    tabular_data['X_test']=normalized_df_test.to_numpy()
    tabular_data['y_valid']=y_valid
    tabular_data['X_valid']=normalized_df_valid.to_numpy()
    tabular_data['y_train_valid']=np.concatenate((tabular_data['y_train'], tabular_data['y_valid']), axis=0)
    tabular_data['X_train_valid']=np.concatenate((tabular_data['X_train'], tabular_data['X_valid']), axis=0)
    
    if no_stratification:
        n_splits=len((df[label_header]))
    else:
        n_splits=len((df_id_class_pos)) - len((df_id_class_pos_test))
                
    return tabular_data, n_splits


def run_experiment(model, model_name, tabular_data, hyperparameter_space, scoring, data_type, n_splits, no_stratification=False):

    result_dict = dict()
    
    X_train = tabular_data['X_train']
    X_train_valid=tabular_data['X_train_valid']
    X_valid=tabular_data['X_valid']
    X_test=tabular_data['X_test']
    
    y_train=tabular_data['y_train']
    y_valid=tabular_data['y_valid']
    y_test=tabular_data['y_test']
    y_train_valid=tabular_data['y_train_valid']
    
    # configure the hyperparameter-tuning procedure
    cv_hyper_parameter_tuner = KFold(n_splits=4, shuffle=True, random_state=1969)
    
    # find best model
    best_model = GridSearchCV(model, hyperparameter_space, scoring=scoring, n_jobs=-1, cv=cv_hyper_parameter_tuner, refit=scoring)
    
    _ = best_model.fit(X_train, y_train)
        
    result_dict[model_name+'_best_params']=best_model.best_params_
    
    y_pred_prob_valid = best_model.predict_proba(X_valid)[:,1]
    
    if scoring=='roc_auc':
        
        result_dict[model_name+'_valid_'+scoring] = metrics.roc_auc_score(y_valid, y_pred_prob_valid)
    else:
        result_dict[model_name+'_valid_'+scoring] = metrics.average_precision_score(y_valid, y_pred_prob_valid)
    
    y_pred_prob_test = best_model.predict_proba(X_test)[:,1]
        
    if scoring=='roc_auc':
        result_dict[model_name+'_test_'+scoring] = metrics.roc_auc_score(y_test, y_pred_prob_test)
    else:
        result_dict[model_name+'_test_'+scoring] = metrics.average_precision_score(y_test, y_pred_prob_test)
    
    result_dict[model_name+'_best_model'] = best_model
    
    print('Best Hyper-parameter Values:')
    
    print(result_dict[model_name+'_best_params'])
    
    print('Validation '+scoring+' Score: '+str(result_dict[model_name+'_valid_'+scoring]))

    # configure the cross-validation procedure
    if no_stratification:
        cv_cross_validation = KFold(n_splits=n_splits, shuffle=True, random_state=1969)
    else:
        cv_cross_validation = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1969)
    # execute the nested cross-validation
    result_dict[model_name+'_cross_validation_'+scoring+'_list'] = cross_val_score(best_model, X_train_valid, y_train_valid, scoring=scoring, cv=cv_cross_validation, n_jobs=-1)
    
    result_dict[model_name+'_cross_validation_'+scoring+'_average'] = mean(result_dict[model_name+'_cross_validation_'+scoring+'_list'])
        
    print('Average cross-validation '+scoring+' score: '+str(result_dict[model_name+'_cross_validation_'+scoring+'_average']))
        
    print('Cross-validation '+scoring+' score list: '+str(result_dict[model_name+'_cross_validation_'+scoring+'_list']))

    print('Test '+scoring+' Score: '+str(result_dict[model_name+'_test_'+scoring]))
    
    print()
    
    f = open("XGBoost_random_forests_tabular_results.csv", "a")
    f.write(model_name+","+data_type+","+str(result_dict[model_name+'_valid_'+scoring])+","+str(result_dict[model_name+'_test_'+scoring])+","+str(result_dict[model_name+'_cross_validation_'+scoring+'_average'])+"\n")
    f.close()
        
    return result_dict


def run_test(best_model, model_name, tabular_data, scoring, data_type, n_splits, no_stratification=False):

    X_train = tabular_data['X_train']
    X_train_valid=tabular_data['X_train_valid']
    X_valid=tabular_data['X_valid']
    X_test=tabular_data['X_test']
    
    y_train=tabular_data['y_train']
    y_valid=tabular_data['y_valid']
    y_test=tabular_data['y_test']
    y_train_valid=tabular_data['y_train_valid']
    
    
    _ = best_model.fit(X_train, y_train)
        
    y_pred_prob_valid = best_model.predict_proba(X_valid)[:,1]
    
    if scoring=='roc_auc':
        
        valid_evaluation_score = metrics.roc_auc_score(y_valid, y_pred_prob_valid)
    else:
        valid_evaluation_score = metrics.average_precision_score(y_valid, y_pred_prob_valid)
        
    
    y_pred_prob_test = best_model.predict_proba(X_test)[:,1]
    
    if scoring=='roc_auc':
        
        test_evaluation_score = metrics.roc_auc_score(y_test, y_pred_prob_test)
    else:
        test_evaluation_score = metrics.average_precision_score(y_test, y_pred_prob_test)
        
    
    print('Validation '+scoring+': '+str(valid_evaluation_score))

    # configure the cross-validation procedure
    if no_stratification:
        cv_cross_validation = KFold(n_splits=n_splits, shuffle=True, random_state=1969)
    else:
        cv_cross_validation = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1969)
    
    # execute the nested cross-validation
    cross_validation_evaluation_score_list = cross_val_score(best_model, X_train_valid, y_train_valid, scoring=scoring, cv=cv_cross_validation, n_jobs=-1)
    
    cross_validation_evaluation_score_average = mean(cross_validation_evaluation_score_list)
        
    print('Average cross-validation '+scoring+' score: '+str(cross_validation_evaluation_score_average))
        
    print('Cross-validation '+scoring+' score list: '+str(cross_validation_evaluation_score_list))

    print('Test '+scoring+' Score: '+str(test_evaluation_score))
    
    print()
    
    f = open("XGBoost_random_forests_tabular_test_results.csv", "a")
    f.write(model_name+","+data_type+","+str(valid_evaluation_score)+","+str(test_evaluation_score)+","+str(cross_validation_evaluation_score_average)+"\n")
    f.close()


def arrange_tabular_data(tabular_data_column_dict):
        
    data_types = list(tabular_data_column_dict.keys())
    
    categorical_column_name_lists = []
    numeric_column_name_lists = []
    
    for data_type in data_types:
        
        categorical_column_name_lists.append(tabular_data_column_dict[data_type][0])
        numeric_column_name_lists.append(tabular_data_column_dict[data_type][1])
    
    data_type_combination_list=[]
    categorical_column_name_combination_lists=[]
    numeric_column_name_combination_lists=[]
    
    index_list = [i for i in range(len(data_types))]
    for L in range(len(index_list) + 1):
        for subset in itertools.combinations(index_list, L):
            combination_index_list = list(subset)
            
            if len(combination_index_list)!=0:
                data_type_combination=""
                categorical_combination=[]
                numeric_combination=[]
                
                for combination_index in combination_index_list:
                    
                    data_type_combination+=data_types[combination_index]
                    categorical_combination+=categorical_column_name_lists[combination_index]
                    numeric_combination+=numeric_column_name_lists[combination_index]
                    
                    if combination_index!=combination_index_list[-1]:
                        data_type_combination+='+'
                
                data_type_combination_list.append(data_type_combination)
                categorical_column_name_combination_lists.append(categorical_combination)
                numeric_column_name_combination_lists.append(numeric_combination)
    
    tabular_column_names_data_type_combination_dict=dict()
    
    for data_type_id in range(len(data_type_combination_list)):
    
        categorical_and_numeric_dict = dict()
    
        categorical_and_numeric_dict['categorical_column_names']=categorical_column_name_combination_lists[data_type_id]
        categorical_and_numeric_dict['numeric_column_names']=numeric_column_name_combination_lists[data_type_id]
    
        tabular_column_names_data_type_combination_dict[data_type_combination_list[data_type_id]]=categorical_and_numeric_dict
    
    return tabular_column_names_data_type_combination_dict, data_type_combination_list


start_time = time.time()

parser = argparse.ArgumentParser()
args = add_args(parser)

patient_file_name = args.patient_file_name
save_dir = args.save_dir
data_type=args.tabular_data_type

is_training_int = args.is_training
is_training = False
if is_training_int!=0:
    is_training=True

#first list is categorical, second one is numerical
tabular_data_column_name_dict= {
    
    'thermal_tabular' : [[], [str(i) for i in range(1, 262)]], # File: LabeledallQuant.xlsx, Sheet: FaceChestBackSides
    
    # File: subject_description.csv
    'symptoms': [['Other Symptom'], ['Fever', 'Cough', 'Sore Throat', 'Diarrhea', 'Vomit', 
                 'Loss of Smell', 'Loss of Taste', 'Chills','Head Aches', 
                    'Muscle Pain', 'Joint Pain', 'Malaise']],
                    
    'vitals' : [[], ['Body Temperature (°C)', 'Blood Pressure S', 'Blood Pressure D', 
                 'Cardiac Frequency', 'SpO2', 'Breathing Frequency', 'Weight (kg)', 'Height(cm)']],
                 
    'drugs' : [['Drug Name', 'Drugs/Vaping intake'],
                ['Drug for Diabetes', 'Drug for Hypertension', 'Drug for Pain', 
                'Drug for Fever', 'Other Drug Used']],
    
    'other' : [['SEX', 'Other Exposure', 'Food Intake'],
            ['AGE', 'Home Exposure', 'Hospital Exposure', 'Work Exposure',
            'Alcohol intake', 'Tobacco smooking', 'Resting an Hour Ago', 'Walking an Hour Ago', 
            'Running an Hour Ago', 'Gym an Hour Ago']]}
            
# choice_list contains all experiments names / tabular_data_type(s) 
data_columns_dict, choice_list = arrange_tabular_data(tabular_data_column_name_dict)

#print(choice_list)



id_header = 'ID'
label_header = 'PCR Result'
positive_text = 'Positive'

scoring = args.evaluation_metric
print("Evaluation Metric: "+scoring)

hyperparameter_spaces=dict()

hyperparameter_spaces['XGBoost'] = {"model__learning_rate": [0.01], 
                                   "model__n_estimators": [250], 
                                   'model__max_depth': [5], 
                                   'model__min_child_weight': [1], 
                                   'model__subsample':[0.5], 
                                   'model__colsample_bytree':[0.9], 
                                   "model__gamma": [i/10.0 for i in range(0,6)], 
                                   "model__reg_lambda": [0, 0.5, 1, 1.5, 2, 3, 4.5]}

hyperparameter_spaces['random_forests'] = {'n_estimators' : [1000, 5000, 10000],
                                          'class_weight' : ['balanced', 'balanced_subsample', None],
                                          'max_features' : ['sqrt', 'log2', None]}


models=dict()
models['XGBoost']=XGBClassifier(eval_metric='logloss', seed=1969, verbosity = 0)
models['random_forests']=RandomForestClassifier(random_state=1969)

tabular_data, n_splits = get_clean_splitted_tabular_data(data_columns_dict, data_type, patient_file_name)

if is_training:

    results_dict = dict()

    if args.is_header==1: #header line
        f = open("XGBoost_random_forests_tabular_results.csv", "a")
        f.write("Model,Data,Validation "+scoring+",Test "+scoring+",Average Stratified Cross-Validation "+scoring+"\n")
        f.close()
    
        
    
    print('Data Type: '+data_type)
    
    model_results_dict = dict()

    for model_name in list(models.keys()):
    
        print(model_name)
    
        model_results_dict[model_name] = run_experiment(models[model_name], model_name, tabular_data, hyperparameter_spaces[model_name], scoring, data_type, n_splits)
        
    results_dict[data_type]=model_results_dict
    
    pickle.dump(results_dict, open(save_dir+data_type+"_XGBoost_random_forests_results.dict", 'wb')) 
    
else: #test only
    
    print('Data Type: '+data_type)
    
    test_results_dict = pickle.load(open(save_dir+data_type+"_XGBoost_random_forests_results.dict", 'rb'))
    
    if args.is_header==1: #header line
        f = open("XGBoost_random_forests_tabular_test_results.csv", "a")
        f.write("Model,Data,Validation "+scoring+",Test "+scoring+",Average Stratified Cross-Validation "+scoring+"\n")
        f.close()

    for model_name in list(models.keys()):
    
        print(model_name)
        
        model=test_results_dict[data_type][model_name][model_name+"_best_model"]
    
        run_test(model, model_name, tabular_data, scoring, data_type, n_splits)
        
print()
print('Finished, Total time taken: {:.0f}m {:.0f}s'.format((time.time() - start_time) // 60,
                                                                    (time.time() - start_time) % 60))


