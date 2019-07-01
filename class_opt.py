"""
#Version 0.201907011700
#Made by Bryan Saldivar
#https://twitter.com/bsaldivaremc2
#Way of usage

import warnings
warnings.filterwarnings("ignore")
classifiers = {
            'logistic_regression':
                    {
                        'init_params':{'random_state':0}
                }
          }

classifiers = {
            'logistic_regression':
                    {
                        'init_params':{'random_state':0},
                        'opt_params':{'C':[0.01,1]}
                },
            'svc':
              {
                   'init_params':{'random_state':0},
                  'opt_params':{'C':[0.01,10],'kernel':['rbf','linear','sigmoid']}
              },
            'gaussian_nb':{},
            'multinomial_nb':{},
            'bernoulli_nb':{},
            'complement_nb':{},
            'ada_boost':{
                'init_params':{'random_state':0},
                 'opt_params':{'n_estimators':[1,10]}
                        },
            'random_forest':{
                'init_params':{'random_state':0},
                'opt_params':{'n_estimators':[1,10],
                             'criterion':['gini','entropy'],
                              'max_depth':[2,10]
                             }
                          }
          }

clf_opt = ClassifierOpt(dfx,dfy,class_col='Y88',test_split=0.25,maximize_metric='accuracy',
        classifiers=classifiers,
            skopt_kargs={'n_calls':10},
            train_weight=1,test_weight=2,repeat=5,cost_method='sum')

clf_opt.start_optimization()

clf_opt.current_clfx = clfs_dic['random_forest'](**{'n_estimators': 8, 'criterion': 'entropy', 'max_depth': 6,'random_state':0})
train_cost, test_cost = clf_opt.simple_cost_function()
print(train_cost,test_cost)

"""

from skopt import gp_minimize
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as logistic_regression
from sklearn.naive_bayes import GaussianNB as gaussian_nb
from sklearn.svm import SVC as support_vector_classifier
from sklearn.naive_bayes import MultinomialNB as multinomial_nb
from sklearn.naive_bayes import BernoulliNB as bernoulli_nb
from sklearn.naive_bayes import ComplementNB as complement_nb
from sklearn.ensemble import AdaBoostClassifier as ada_boost
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import RandomForestClassifier as random_forest
metrics_dic = {'accuracy':accuracy}
clfs_dic = {
            'logistic_regression':logistic_regression,
            'gaussian_nb':gaussian_nb,
            'svc':support_vector_classifier,
            'multinomial_nb':multinomial_nb,
            'bernoulli_nb':bernoulli_nb,
            'complement_nb':complement_nb,
            'ada_boost':ada_boost,
            'random_forest':random_forest
        }

def split_train_test(idf, class_col,test_split=0.25):
  labels = list(idf[class_col].unique())
  test_index = []
  train_index = []
  for label in labels:
    tdf = idf[idf[class_col]==label]
    n = tdf.shape[0]
    n_test = int(np.ceil(n*test_split))
    indexes = list(tdf.index.values)
    test_i = np.random.choice(indexes,n_test,replace=False)
    train_i = list(filter(lambda x: x not in test_i,indexes))
    train_index.extend(train_i)
    test_index.extend(test_i)
  return train_index[:],test_index[:]



class ClassifierOpt:
    def __init__(self,dfx,dfy,
    class_col,test_split=0.25,maximize_metric='accuracy',
        classifiers={
            'logistic_regression':
                    {
                        'init_params':{},
                        'opt_params':{}
                    }
                },
            skopt_kargs={'n_calls':10},
            train_weight=1,test_weight=2,repeat=5,cost_method='sum'
        ):
        self.dfx = dfx.copy()
        self.dfy = dfy.copy()
        self.class_col = class_col
        self.X = self.dfx.values.astype('float32')
        self.y = self.dfy[self.class_col].values.astype(int)
        self.test_split = test_split
        self.maximize_metric = metrics_dic.get(maximize_metric,'accuracy')
        self.clfs=classifiers.copy()
        self.skopt_kargs=skopt_kargs.copy()
        self.train_weight=train_weight
        self.test_weight=test_weight
        self.repeat = repeat
        self.cost_method=cost_method
    def init_clfs(self):
        pass
    def simple_cost_function(self):
        train_index, test_index = split_train_test(self.dfy, self.class_col,self.test_split)
        self.X_train,self.y_train = self.X[train_index],self.y[train_index]
        self.X_test,self.y_test = self.X[test_index],self.y[test_index]
        #Fit
        self.current_clfx.fit(self.X_train,self.y_train)
        self.train_pred = self.current_clfx.predict(self.X_train)
        self.test_pred = self.current_clfx.predict(self.X_test)
        self.train_metric = self.maximize_metric(self.y_train,self.train_pred)
        self.test_metric = self.maximize_metric(self.y_test,self.test_pred)
        return (self.train_metric,self.test_metric)
    def cost_function(self,input_hyper_parameters):
        self.current_opt_kargs =  {}
        for name,value in zip(self.current_clf_opt_names,input_hyper_parameters):
            self.current_opt_kargs[name]=value
        self.current_opt_kargs.update(self.current_clf_init_params)
        self.current_clfx = clfs_dic[self.current_clf_name](**self.current_opt_kargs)
        #For repeat
        costs = []
        for repeat_iter in range(self.repeat):
            self.simple_cost_function()
            costs.append(
                (1-self.train_metric)*self.train_weight + (1-self.test_metric)*self.test_weight)
        if self.cost_method=='sum':
            cost = np.sum(costs)
        elif self.cost_method=='mean':
            cost = np.mean(costs)
        elif self.cost_method=='max':
            cost = np.max(costs)
        return cost
    def optimize(self):
        best_name_value = {}
        if len(self.current_clf_opt_params)>0:
            self.optimizer = gp_minimize(self.cost_function,self.current_clf_opt_values,**self.skopt_kargs)
            for name, value in zip(self.current_clf_opt_names,self.optimizer.x):
                best_name_value[name]=value
            self.current_clf_metric = self.optimizer.fun
            self.current_clf_optimized = best_name_value.copy()
        else:
            self.current_clf_metric = self.cost_function([])
            self.current_clf_optimized = {}
    def get_names_values(self,idic):
        names = []
        values = []
        for k in idic.keys():
            names.append(k)
            values.append(idic[k][:])
        return names[:],values[:]
    def start_optimization(self):
        self.init_clfs()
        for clf_name in self.clfs.keys():
            self.current_clf_name = clf_name
            clf = self.clfs[clf_name]
            self.current_clf = clf
            self.current_clf_init_params=clf.get('init_params',{})
            self.current_clf_opt_params=clf.get('opt_params',{})
            self.current_clf_opt_names,self.current_clf_opt_values = self.get_names_values(self.current_clf_opt_params)
            self.optimize()
            print(self.current_clf_name)
            print(self.current_clf_metric)
            print(self.current_clf_optimized)
            print("")
        #
