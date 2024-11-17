#!/usr/bin/env python
# coding: utf-8
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import time
import argparse
from copy import deepcopy
from numpy import mean
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

def add_args(parser):

    parser.add_argument('--tabular_data_type', type=str,
                        help='tabular data type')
    
    parser.add_argument('--trial_num_in_total', type=int, default=100,
                        help='number of total trials for hyperparameter tuning')
    
    parser.add_argument('--batch_size', type=int, default=0,
                        help='mini-batch size, 0 means it will take the whole train datasize')
    
    parser.add_argument('--patient_file_name', type=str, default='/home/dsv/maul8511/Desktop/paper-8/final/all_data.csv',
                        help='tabular data file')
    
    parser.add_argument('--save_dir', type=str, default='',
                        help='result directory')
    
    parser.add_argument('--cross_validation', type=int, default=1,
                        help='--cross_validation')
    
    parser.add_argument('--is_training', type=int, default=1,
                        help='training or testing')
    
    parser.add_argument('--is_transform', type=int, default=1,
                        help='transforming the data')
                        
    parser.add_argument('--is_header', type=int, default=0,
                        help='writing the header line')
    
    parser.add_argument('--evaluation_metric', type=str, default='roc_auc',
                        help='evaluation metric, roc_auc, average_precision')
    
    args = parser.parse_args()
    
    return args


def get_clean_splitted_tabular_data(data_columns_dict, choice, data_file_name, transform, cross_validation=False, no_stratification=False, id_header='ID', label_header='PCR Result', positive_text='Positive'):
    
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

    df = df[selected_column_names+[label_header]]
                
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
    
    tabular_dfs = dict()
    
    if transform:
    
        scaler = StandardScaler()
           
        normalized_df_train = pd.DataFrame(
        scaler.fit_transform(df_train[selected_column_names]),
        columns = selected_column_names
        )
    
        normalized_df_test = pd.DataFrame(
        scaler.transform(df_test[selected_column_names]),
        columns = selected_column_names
        )
    
        normalized_df_valid = pd.DataFrame(
        scaler.transform(df_valid[selected_column_names]),
        columns = selected_column_names
        )
    
        normalized_df_train[label_header]=list(df_train[label_header])
        normalized_df_test[label_header]=list(df_test[label_header])
        normalized_df_valid[label_header]=list(df_valid[label_header])
        
        train_data=normalized_df_train
        test_data=normalized_df_test
        valid_data=normalized_df_valid
    else:
        train_data=df_train
        test_data=df_test
        valid_data=df_valid
        
    tabular_dfs['train']=train_data
    tabular_dfs['test']=test_data
    tabular_dfs['valid']=valid_data
    
    if no_stratification:
        n_splits=len((df[label_header]))
    else:
        n_splits=len((df_id_class_pos)) - len((df_id_class_pos_test))
        
    if cross_validation:
        
        df_train_valid_cv = pd.concat([train_data, valid_data])
        
        cross_validation_data=dict()
        
        if no_stratification:
            cv_cross_validation = KFold(n_splits=n_splits, shuffle=True, random_state=1969)
        else:
            cv_cross_validation = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1969)
            
        X=df_train_valid_cv[selected_column_names]
        y=df_train_valid_cv[label_header]
            
        for i, (train_index, valid_index) in enumerate(cv_cross_validation.split(X, y)):
            cross_validation_data['fold_'+str(i)+'_train']=df_train_valid_cv.iloc[train_index]
            cross_validation_data['fold_'+str(i)+'_valid']=df_train_valid_cv.iloc[valid_index]
            
        tabular_dfs['cross_validation_data']=cross_validation_data
            
    return tabular_dfs, len(selected_column_names), n_splits


class tabular_Dataset(Dataset):

    def __init__(self, df):
    
        self.y = np.array([0 if label == 0 else 1 for label in df['PCR Result']])
        self.all_tabular_features = df.drop(columns=['PCR Result'], axis=1).values.tolist()
        
    def __getitem__(self, index):
              
        tabular_features = np.array(self.all_tabular_features[index])
        
        label = self.y[index]
                
        return tabular_features, label

    def __len__(self):
        return self.y.shape[0]


class MLP_Tabular(nn.Module):
    
    
    def __init__(self, neural_network_parameters):
        super(MLP_Tabular, self).__init__() # Initialize parent class
        
        device=neural_network_parameters['device']

        tab_fc_layers = []

        tabular_in_neurons = neural_network_parameters['total_tabular_input_features']
    
        for i in range(neural_network_parameters['num_tabular_fc_layers']):
                
            tab_fc_layers.append(nn.Linear(tabular_in_neurons, neural_network_parameters['tabular_fc_neurons_list'][i]).to(device))
        
            tab_fc_layers.append(nn.ReLU())
        
            tabular_in_neurons = neural_network_parameters['tabular_fc_neurons_list'][i]
            
        self.fcs_for_tabs = nn.Sequential(*tab_fc_layers)
        
        self.final_layer = nn.Linear(neural_network_parameters['tabular_fc_neurons_list'][-1], neural_network_parameters['num_classes']).to(device)
        
        self.relu = nn.ReLU()
        
    def forward(self, tab):
        
        tab=tab.float()
        
        for i, fc_tab_i in enumerate(self.fcs_for_tabs):
            
            tab = fc_tab_i(tab)
                
        final = self.final_layer(tab)
        
        logits = self.relu(final)
        
        probas = torch.softmax(logits, dim=1)
        
        return logits, probas


def train_model(model, optimizer, train_dataloader, val_dataloader, num_epochs, evaluation_metric, trial=None):
    
    torch.manual_seed(1)
    
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.train()
    
    for epoch in range(num_epochs):
        
        for batch_idx, (features, targets) in enumerate(train_dataloader):
        
            features = features.to(device)
            targets = targets.to(device)

            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
        
            cost.backward()
        
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
        
            ### LOGGING
            #if not batch_idx % 60:
            #    print ('Epoch: %01d/%01d | Batch %02d/%02d | Cost: %.4f' 
            #       %(epoch+1, num_epochs, batch_idx, 
            #         len(train_dataloader), cost))
            
            #if batch_idx==0:
            #    break
            
        evaluation_score = evaluation(model, val_dataloader, evaluation_metric)
            
        if trial!=None:
            # Add prune mechanism
            trial.report(evaluation_score, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
    return deepcopy(model.state_dict()), evaluation_score
    

def evaluation(model, data_loader, evaluation_metric):
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct_pred, num_examples = 0, 0
    y_true_labels = []
    y_predicted_probability_scores = []
    y_predicted_labels = []
    for features, targets in data_loader:
        
        features = features.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():

            logits, probas = model(features)
        
        proba_score, predicted_labels = torch.max(probas, 1)
        
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
        y_true_labels.extend(targets.tolist())
        y_predicted_probability_scores.extend(proba_score.tolist())        
        y_predicted_labels.extend(predicted_labels.tolist())
        
    
    if evaluation_metric=='roc_auc':
        
        try:
            evaluation_score = roc_auc_score(y_true_labels,  y_predicted_probability_scores)
        except ValueError:
            evaluation_score = 0
    else:  
        try:
            evaluation_score = average_precision_score(y_true_labels,  y_predicted_probability_scores)
        except ValueError:
            evaluation_score = 0
        
    return evaluation_score


def objective(trial, total_tabular_input_features, train_dataloader, val_dataloader, save_name, evaluation_metric):
    
    """
    An objective function that accepts multiple parameters.
    """
    
    neural_network_parameters = dict()
    
    
    neural_network_parameters['num_tabular_fc_layers'] = trial.suggest_int("num_tabular_fc_layers", 2, 3)
    
    neural_network_parameters['num_epochs']=trial.suggest_int("num_epochs", 2, 3)
    
    neural_network_parameters['tabular_fc_neurons_list']=[trial.suggest_int("tabular_neurons_layer_{}".format(i), int(total_tabular_input_features*1.2), int(total_tabular_input_features*1.5))
                   for i in range(neural_network_parameters['num_tabular_fc_layers'])]
    
    
    neural_network_parameters['optimizer']=trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    
    neural_network_parameters['lr']=trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    neural_network_parameters['device']=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_network_parameters['total_tabular_input_features']=total_tabular_input_features
    neural_network_parameters['num_classes']=2
    
    # Generate the model
    model = MLP_Tabular(neural_network_parameters)
    
    # Generate the optimizers
    optimizer = getattr(optim, neural_network_parameters['optimizer'])(model.parameters(), lr=neural_network_parameters['lr'])
    
    trained_model_state_dict, evaluation_score = train_model(model, optimizer, train_dataloader, val_dataloader, neural_network_parameters['num_epochs'], evaluation_metric, trial=trial)
    
    trial.set_user_attr(key="current_model_state_dict", value=trained_model_state_dict)

    trial.set_user_attr(key="save_name", value=save_name)
    
    trial.set_user_attr(key="total_tabular_input_features", value=total_tabular_input_features)
    
    return evaluation_score


def callback(study, trial):
    
    if study.best_trial.number == trial.number:
         
        study.set_user_attr(key="best_model_state_dict", value=trial.user_attrs["current_model_state_dict"])
        
        current_best_model_state_dict = trial.user_attrs["current_model_state_dict"]
        
        current_best_model_parameters = study.best_trial.params
        
        current_best_model_save_name = trial.user_attrs["save_name"]
        
        torch.save(current_best_model_state_dict, current_best_model_save_name)
        
        total_tabular_input_features = trial.user_attrs["total_tabular_input_features"]
                   
        neural_network_parameters = arrange_NN_parameters(current_best_model_parameters, total_tabular_input_features)

        pickle.dump(neural_network_parameters, open(current_best_model_save_name+'_hyper_parameters', 'wb'))
        

def calculate_cross_validation(data_type, no_stratification, neural_network_parameters, data_columns_dict, patient_file_name, batch_size, save_dir, transform, evaluation_metric):
    
    
    cross_validation_tabular_dfs, total_tabular_input_features, n_splits = get_clean_splitted_tabular_data(data_columns_dict, data_type, patient_file_name, transform, cross_validation=True, no_stratification=no_stratification)
    
    cross_validation_evaluation_score_list = []

    for fold in range(n_splits):
    
        reset_model = MLP_Tabular(neural_network_parameters)
    
        cross_validation_best_optimizer = getattr(optim, neural_network_parameters['optimizer'])(reset_model.parameters(), lr=neural_network_parameters['lr'])
    
        cross_validation_train = tabular_Dataset(cross_validation_tabular_dfs['cross_validation_data']['fold_'+str(fold)+'_train'])
        cross_validation_val = tabular_Dataset(cross_validation_tabular_dfs['cross_validation_data']['fold_'+str(fold)+'_valid'])
       
        cross_validation_train_dataloader = DataLoader(cross_validation_train, batch_size=batch_size, drop_last=False)
        cross_validation_val_dataloader = DataLoader(cross_validation_val, batch_size=batch_size)
    
        cross_validation_trained_model_state_dict, evaluation_score = train_model(reset_model, cross_validation_best_optimizer, cross_validation_train_dataloader, cross_validation_val_dataloader, neural_network_parameters['num_epochs'], evaluation_metric)
    
        cross_validation_evaluation_score_list.append(evaluation_score)
        
        save_model_name = save_dir+"mlp_cross_validation_model_fold_"+str(fold+1)+"_no_stratification_"+str(no_stratification)+"_"+data_type+"_batch_size_"+str(batch_size)+".pt"

        torch.save(cross_validation_trained_model_state_dict, save_model_name)
        
    return cross_validation_evaluation_score_list
    

def arrange_NN_parameters(model_hyperparameters, total_tabular_input_features):
    
    neural_network_parameters = dict()
    
    neural_network_parameters['num_tabular_fc_layers']=model_hyperparameters['num_tabular_fc_layers']
    neural_network_parameters['num_epochs']=model_hyperparameters['num_epochs']
    neural_network_parameters['device']=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_network_parameters['tabular_fc_neurons_list']=[model_hyperparameters['tabular_neurons_layer_'+str(i)]
                   for i in range(neural_network_parameters['num_tabular_fc_layers'])]
    neural_network_parameters['optimizer']=model_hyperparameters['optimizer']
    neural_network_parameters['lr']=model_hyperparameters['lr']
    neural_network_parameters['total_tabular_input_features']=total_tabular_input_features
    neural_network_parameters['num_classes']=2
    
    return neural_network_parameters


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

cross_validation = args.cross_validation

is_training_int = args.is_training

is_training = False

if is_training_int!=0:
    is_training=True

id_header = 'ID'
label_header = 'PCR Result'
positive_text = 'Positive'

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

data_type=args.tabular_data_type

transform=False
if args.is_transform!=0:
    transform=True

print('Data Type: '+data_type)

batch_size = args.batch_size
print("Batch Size: "+str(batch_size))

evaluation_metric = args.evaluation_metric
print("Evaluation Metric: "+evaluation_metric)

n_trials=args.trial_num_in_total
print('Total Hyperparameter training trials: ' +str(n_trials))

tabular_dfs, total_tabular_input_features, n_splits = get_clean_splitted_tabular_data(data_columns_dict, data_type, patient_file_name, transform)

if batch_size == 0:
    batch_size = len(tabular_dfs['train'])

train = tabular_Dataset(tabular_dfs['train'])
val = tabular_Dataset(tabular_dfs['valid'])
test = tabular_Dataset(tabular_dfs['test'])

train_dataloader = DataLoader(train, batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test, batch_size=batch_size)
val_dataloader = DataLoader(val, batch_size=batch_size)

save_name=save_dir+'mlp_current_best_model_'+data_type+'_batch_size_'+str(batch_size)+".pt"
results_dict= dict()

if is_training:
    
    # Wrap the objective inside a lambda and call objective inside it
    objective_func = lambda trial: objective(trial, total_tabular_input_features, train_dataloader, val_dataloader, save_name, evaluation_metric)

    # Pass func to Optuna studies
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective_func, callbacks=[callback], n_trials=n_trials, gc_after_trial=True)
    
    study_best_trial = study.best_trial

    neural_network_parameters = arrange_NN_parameters(study_best_trial.params, total_tabular_input_features)

    results_dict['neural_network_parameters']=neural_network_parameters

    print('Best Hyper-parameter Values:')

    for key, value in study_best_trial.params.items():
        print("{}: {}".format(key, value))
    
    best_model = MLP_Tabular(neural_network_parameters)
    best_model_state_dict = study.user_attrs["best_model_state_dict"]
    best_model.load_state_dict(best_model_state_dict)
    
    torch.save(best_model_state_dict, save_dir+"mlp_fine_tuned_best_model_"+data_type+"_batch_size_"+str(batch_size)+".pt")

else: #test only
    
    neural_network_parameters = pickle.load(open(save_name+'_hyper_parameters', 'rb'))
    best_model = MLP_Tabular(neural_network_parameters)
    best_model.load_state_dict(torch.load(save_name))
    
validation_evaluation_score=evaluation(best_model, val_dataloader, evaluation_metric)

print('Validation '+evaluation_metric+' Score: '+str(validation_evaluation_score))

test_evaluation_score=evaluation(best_model, test_dataloader, evaluation_metric)

print('Test '+evaluation_metric+' Score: '+str(test_evaluation_score))

results_dict[data_type+"_test_"+evaluation_metric]=test_evaluation_score

results_dict[data_type+"_validation_"+evaluation_metric]=validation_evaluation_score

if cross_validation!=0:
    
    cv_with_evaluation_score_list = calculate_cross_validation(data_type, False, neural_network_parameters, data_columns_dict, patient_file_name, batch_size, save_dir, transform, evaluation_metric)
    mean_with_cv=mean(np.array(cv_with_evaluation_score_list))
    
    print('With stratification, Average cross-validation '+evaluation_metric+' score: '+str(mean_with_cv))
    print('With stratification, Cross-validation '+evaluation_metric+' score list: '+str(cv_with_evaluation_score_list))
    
    f = open("mlp_results.csv", "a")
    if args.is_header==1: #header line
        f.write("Model,Data,Batch Size,Validation "+evaluation_metric+",Test "+evaluation_metric+",Average Stratified Cross-Validation "+evaluation_metric+"\n")
    f.write("MLP,"+data_type+","+str(validation_evaluation_score)+","+str(test_evaluation_score)+","+str(mean_with_cv)+"\n")
    f.close()
         
    results_dict[data_type+"_cv_with_stratification_avg_"+evaluation_metric+"_score"]=mean_with_cv
    results_dict[data_type+"_cv_with_stratification_"+evaluation_metric+"_score_list"]=cv_with_evaluation_score_list
    
print(results_dict)

if is_training:

    pickle.dump(results_dict, open(save_dir+"mlp_training_results_dict_"+data_type+"_batch_size_"+str(batch_size), 'wb'))
    
else:
    
    pickle.dump(results_dict, open(save_dir+"mlp_test_results_dict_"+data_type+"_batch_size_"+str(batch_size), 'wb'))
    
    
print()
print('Finished, Total time taken: {:.0f}m {:.0f}s'.format((time.time() - start_time) // 60,
                                                                    (time.time() - start_time) % 60))

