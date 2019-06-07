"""
Version 0.201904261100
Created by Bryan saldivar
https://twitter.com/bsaldivaremc2
#How to use:
#Simple usage:
target_labels = ddf_cols
regressors = {
              'linear_regression':{},
              'ridge':{
                       'init_params':{'random_state':0},
                       'opt_params':{'alpha':[0.01,1.0]}
                      },
              'lasso':{
                       'init_params':{'random_state':0},
                       'opt_params':{'alpha':[0.01,1.0]}
                       },
              'multi_task_lasso':{
                       'init_params':{'random_state':0},
                       'opt_params':{'alpha':[0.01,1.0]}
                       },
              'elastic_net':{
                       'init_params':{'random_state':0},
                       'opt_params':{'alpha':[0.01,1.0],'l1_ratio':[0.01,1.0]}
                       },
              'multi_task_elastic_net':{
                       'init_params':{'random_state':0},
                       'opt_params':{'alpha':[0.01,1.0],'l1_ratio':[0.01,1.0]}
                       },
              'lasso_lars':{
                       'init_params':{},
                       'opt_params':{'alpha':[0.01,1.0]}
                       },
              'orthogonal_matching_pursuit':{},
             }

MFR = MFeatureRegressor(X,y_no_cat,regressors,target_labels=target_labels,
        test_split = 0.25, repeat=5,train_weight=1,test_weight=4,skopt_kargs={'n_calls':10},v=False)

MFR.start_optimization()

MFR.get_train_test() #Gets a new train/test split 
MFR.get_predictor() #Mixes all the regressors with the best hyperparameters
pred = MFR.predict(MFR.x_test) # Make a prediction with a given input
#MFR.get_test_pred_diff() #Find the absolute difference between the testing set and the prediction, difference normalized
#by the statistics from the training set
MFR.get_test_train_diff() #Fin the absolute difference between the testing and the training set picking a random sample from 
#the training set
#MFR.test_pred_diff_norm_mean #A variable generated by the process above

##More sophisticated usage:

#Try multithreading with multiple objects
frs = []
for _ in range(y.shape[1]):
    frs.append(
        FeatureRegressor(X,y[:,_:_+1],regressors=regressors,target_labels=[target_labels[_]],
        test_split = 0.25, repeat=1000,train_weight=1,test_weight=4,skopt_kargs={'n_calls':10})
    )

import _thread
def fr_x(frindex):
    frs[frindex].start_optimization()

for _ in range(len(frs)):
    _thread.start_new_thread(fr_x,(_,) )

total_regs = len(list(regressors.keys()))
for _ in range(len(frs)):
    xvar = list(frs[_].report.keys())
    done = len(list(frs[_].report[xvar[0]].keys()))
    print(xvar[0],done,"/",total_regs)

reports = []
for _ in range(len(frs)):
    reports.append(frs[_].best_report.copy())

report_df = []
for _ in range(len(reports)):
    name = list(reports[_].keys())[0]
    _ = reports[_][name]
    value = 1000 - _['value']
    reg_name = _['name']
    params = _['params']
    report_df.append({'variable':name,'value':value,'reg_name':reg_name,'params':params})

report_df = pd.DataFrame(report_df)

"""
from sklearn.linear_model import LinearRegression,Ridge,Lasso,MultiTaskLasso,ElasticNet,MultiTaskElasticNet,LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
import scipy
from skopt import gp_minimize
import skopt
from skopt.space import Real, Integer, Categorical
import numpy as np
import pandas as pd
def norm_all(train,test,pred_test,mode='1_to_1'):
    def norm_std_mean(inp,std,mean):
        o = (inp-mean)/std
        return o.copy()
    m = train.max(0)
    mi = train.min(0)
    mean = train.mean(0)
    std = train.std(0)
    if mode=='1_to_1':
        std = m - mi
        mean = mi
    _train = norm_std_mean(train,std,mean)
    _test = norm_std_mean(test,std,mean)
    _pred_test = norm_std_mean(pred_test,std,mean)
    if mode=='1_to_1':
        _train = _train*2 - 1
        _test = _test*2 - 1
        _pred_test = _pred_test*2 - 1
    return _train.copy(),_test.copy(),_pred_test.copy()
class MFeatureRegressor:
    def __init__(self,ix,iy,regressors,target_labels=[],
        test_split = 0.25, repeat=5,train_weight=1,test_weight=2,skopt_kargs={'n_calls':10},v=False,
        mse_min_val='mean',minimize_value='p'):
        """
            regressors: a dictionary with the form:
            regressors = {'linear_regression':{},
                            'ridge':{'init_params':{'random_state':0},
                                    'opt_params':{'alpha':[0.01,1.0]}
                                    }
                            }
            mse_min_val: Method to get the mse:
                available values= 'mean','min','max'
        #Simple usage:
            target_labels = ddf_cols
            regressors = {
                          'linear_regression':{},
                          'ridge':{
                                   'init_params':{'random_state':0},
                                   'opt_params':{'alpha':[0.01,1.0]}
                                  },
                          'lasso':{
                                   'init_params':{'random_state':0},
                                   'opt_params':{'alpha':[0.01,1.0]}
                                   },
                          'multi_task_lasso':{
                                   'init_params':{'random_state':0},
                                   'opt_params':{'alpha':[0.01,1.0]}
                                   },
                          'elastic_net':{
                                   'init_params':{'random_state':0},
                                   'opt_params':{'alpha':[0.01,1.0],'l1_ratio':[0.01,1.0]}
                                   },
                          'multi_task_elastic_net':{
                                   'init_params':{'random_state':0},
                                   'opt_params':{'alpha':[0.01,1.0],'l1_ratio':[0.01,1.0]}
                                   },
                          'lasso_lars':{
                                   'init_params':{},
                                   'opt_params':{'alpha':[0.01,1.0]}
                                   },
                          'orthogonal_matching_pursuit':{},
                         }

            MFR = MFeatureRegressor(X,y_no_cat,regressors,target_labels=target_labels,
                    test_split = 0.25, repeat=5,train_weight=1,test_weight=4,skopt_kargs={'n_calls':10},v=False)

            MFR.start_optimization()

            MFR.get_train_test() #Gets a new train/test split 
            MFR.get_predictor() #Mixes all the regressors with the best hyperparameters
            pred = MFR.predict(MFR.x_test) # Make a prediction with a given input
            #MFR.get_test_pred_diff() #Find the absolute difference between the testing set and the prediction, difference normalized
            #by the statistics from the training set
            MFR.get_test_train_diff() #Fin the absolute difference between the testing and the training set picking a random sample from 
            #the training set
            #MFR.test_pred_diff_norm_mean #A variable generated by the process above
        """
        self.available_regressors = {'linear_regression':LinearRegression,'ridge':Ridge,'lasso':Lasso,
        'multi_task_lasso':MultiTaskLasso,'elastic_net':ElasticNet,'multi_task_elastic_net':MultiTaskElasticNet,
        'lasso_lars':LassoLars,'orthogonal_matching_pursuit':OrthogonalMatchingPursuit}
        self.X = ix
        self.y = iy
        self.validate_y()
        self.regressors = regressors.copy()
        self.target_labels = target_labels[:]
        self.test_split = test_split
        self.repeat = repeat
        self.train_weight = train_weight
        self.test_weight = test_weight
        self.skopt_kargs = skopt_kargs.copy()
        self.total_features = self.y.shape[1]
        self.yi = self.y[:,0:1]
        self.best_params = {}
        self.v=v
        self.mse_min_val = mse_min_val
        self.minimize_value = minimize_value
        self.validate_regressors()
        self.current_regressor_name = None
        self.current_regressor = None
        self.current_regressor_params = {}
        self.current_regressor_init_params = None
        self.current_opt_params = None
        self.current_opt_params_values = []
        self.current_opt_params_names = []
        self.current_target_index = 0
        self.current_target_label = None
        self.report = {}
        self.best_report = []
        self.sorted_report = []
        self.optimizer = None
        self.current_opt_call = 0
    def validate_y(self):
        if len(self.y.shape)==1:
            print("y should have a number of columns at least of 1. Reshaping")
            self.y = self.y.reshape(self.y.shape[0],1)
    def validate_regressors(self):
        if type(self.regressors)==dict:
            valid_keys = self.available_regressors.keys()
            for k in self.regressors.keys():
                if k not in valid_keys:
                    print(k,"is not a valid regressor.")
                    print("Use any of the following:",valid_keys)
                    break
        else:
            print("regressor should be dictionary. use help(this function) to see the format")
            
    def init_regressor(self):
        self.current_regressor_params = {}
        self.current_opt_params_values = []
        self.current_opt_params_names = []
        self.current_regressor = self.available_regressors[self.current_regressor_name]
        regresor_info = self.regressors[self.current_regressor_name]
        if type(regresor_info)!=dict:
            self.current_regressor_init_params = {}
            self.current_opt_params = {}
        else:
            if 'init_params' not in regresor_info.keys():
                self.current_regressor_init_params = {}
            else:
                self.current_regressor_init_params = regresor_info['init_params']
            if 'opt_params' not in regresor_info.keys():
                self.current_regressor_opt_params = {}
            else:
                self.current_regressor_opt_params = regresor_info['opt_params']
        self.current_regressor_params.update(self.current_regressor_init_params)
        for param_name in self.current_regressor_opt_params.keys():
            self.current_opt_params_names.append(param_name)
            self.current_opt_params_values.append(self.current_regressor_opt_params[param_name])
        #Create spaces
        self.current_opt_call = 0
    
    def cross_val_val(self):
        def predict_mse_val(ireg,ix,iy):
            _pred = ireg.predict(ix)
            t, p = scipy.stats.ttest_rel(iy,_pred, axis=0)
            p_bool = p>0.05
            if self.mse_min_val=='mean':
                mse = np.power(iy-_pred,2).mean(0)
            if self.mse_min_val=='min':
                mse = np.power(iy-_pred,2).min(0)
            if self.mse_min_val=='max':
                mse = np.power(iy-_pred,2).max(0)
            return mse.copy(),p_bool.copy()
        mse_all = []
        p_all = []
        n = self.X.shape[0]
        test_n = int(np.ceil(n*self.test_split))
        train_n = n - test_n
        n_list = list(np.arange(n))
        train_mse = []
        test_mse = []
        for repeatx in range(self.repeat):
            #Change this to a numpy evaluation
            test_index = np.random.choice(n_list,test_n,replace=False)
            train_index = list(filter(lambda x: x not in test_index,n_list))
            xtr,xts = self.X[train_index,:],self.X[test_index,:]
            ytr,yts = self.yi[train_index,:],self.yi[test_index,:]
            _reg = self.current_regressor(**self.current_regressor_params)
            _reg.fit(X=xtr,y=ytr)
            mse_tr,p_tr = predict_mse_val(_reg,xtr,ytr)
            mse_ts,p_ts = predict_mse_val(_reg,xts,yts)
            train_mse.append(mse_tr.copy())
            test_mse.append(mse_ts.copy())
            p_tr_ts = list(map(lambda x,y: int(x and y),p_tr,p_ts))
            p_all.append(p_tr_ts[:])
        train_mse = np.vstack(train_mse).mean(0)
        test_mse = np.vstack(test_mse).mean(0)
        p_all = np.vstack(p_all).sum(0)
        if self.v==True:
            print(p_all)
        #
        if self.minimize_value=='p':
            output_value = self.repeat - p_all.min()
        if self.minimize_value=='mse':
            output_value = train_mse.mean()*self.train_weight+test_mse.mean()*self.test_weight
        self.report[self.current_regressor_name][self.current_opt_call]['val'] = output_value
        self.report[self.current_regressor_name][self.current_opt_call]['p'] = p_all
        self.report[self.current_regressor_name][self.current_opt_call]['mse_train'] = train_mse.copy()
        self.report[self.current_regressor_name][self.current_opt_call]['mse_test'] = test_mse.copy()
        return output_value
    
    def cross_val_metrics(self):
        def predict_mse_val(ireg,ix,iy):
            _pred = ireg.predict(ix)
            t, p = scipy.stats.ttest_rel(iy,_pred, axis=0)
            p_bool = p>0.05
            mse = np.power(iy-_pred,2).sum()
            return mse.copy(),p_bool.copy()
        mse_all = []
        p_all = []
        n = self.X.shape[0]
        test_n = int(np.ceil(n*self.test_split))
        train_n = n - test_n
        n_list = list(np.arange(n))
        for repeatx in range(self.repeat):
            #Change this to a numpy evaluation
            test_index = np.random.choice(n_list,test_n,replace=False)
            train_index = list(filter(lambda x: x not in test_index,n_list))
            xtr,xts = self.X[train_index,:],self.X[test_index,:]
            ytr,yts = self.yi[train_index,:],self.yi[test_index,:]
            _reg = self.current_regressor(**self.current_regressor_params)
            _reg.fit(X=xtr,y=ytr)
            mse_tr,p_tr = predict_mse_val(_reg,xtr,ytr)
            mse_ts,p_ts = predict_mse_val(_reg,xts,yts)
            mse_all.append(self.train_weight*mse_tr+self.test_weight*mse_ts)
            p_tr_ts = list(map(lambda x,y: int(x and y),p_tr,p_ts))
            p_all.append(p_tr_ts[:])
        mse_all = np.sum(mse_all)
        p_all = np.vstack(p_all).sum(0).min()
        if self.v==True:
            print(p_all)
        #
        return self.repeat - p_all

    def cross_val_rep(self,opt_params):
        self.current_opt_call += 1
        self.report[self.current_regressor_name][self.current_opt_call]={}
        for name,value in zip(self.current_opt_params_names,opt_params):
            self.current_regressor_params[name]=value
        #Validate if params exist in current regressor
        self.report[self.current_regressor_name][self.current_opt_call]['params']=self.current_regressor_params.copy()
        return self.cross_val_val()

    def optimize_regressor(self):
        from skopt import gp_minimize
        best_name_value = {}
        if len(self.current_opt_params_values)>0:
            self.optimizer = gp_minimize(self.cross_val_rep,self.current_opt_params_values,**self.skopt_kargs)
        else:
            self.report[self.current_regressor_name][self.current_opt_call]={}
            self.report[self.current_regressor_name][self.current_opt_call]['params']=self.current_regressor_params.copy()
            value_function = self.cross_val_val()
        if self.v==True:
            print(self.report)
    def start_optimization(self):
        if len(self.y.shape)==1:
            self.y = self.y.reshape(self.y.shape[0],1)
        self.yi = self.y[:,:]
        for regressor_name in self.regressors.keys():
            self.current_regressor_name=regressor_name
            self.report[self.current_regressor_name]={}
            self.init_regressor()
            self.optimize_regressor()
        #
        print("Optimization Done. Generating reports")
        if len(self.target_labels)<self.y.shape[1]:
            self.target_labels = [ "feature_"+str(_) for _ in range(self.y.shape[1])]
        data = []
        for regn in self.report.keys():
            rr = self.report[regn]
            for calln in rr.keys():
                call = rr[calln]
                valx = call['val']
                params = call['params']
                mse_tr = call['mse_train']
                mse_ts = call['mse_test']
                ps = call['p']
                for feature in range(len(self.target_labels)):
                    p = ps[feature]
                    mtr = mse_tr[feature]
                    mts = mse_ts[feature]
                    data.append({'Variable':self.target_labels[feature],'regressor':regn,'calln':calln,'params':params,'p':p,'mse_train':mtr,'mse_test':mts})
        report_df = pd.DataFrame(data)
        report_df['inverse_mse_test']=report_df['mse_test'].max()-report_df['mse_test']
        report_df['inverse_mse_train']=report_df['mse_train'].max()-report_df['mse_train']
        feats = report_df['Variable'].unique()
        best_report = []
        sorted_report = []
        for feat in feats:
            sr = report_df[report_df['Variable']==feat].sort_values(by=['p','inverse_mse_test','inverse_mse_train'],ascending=False).reset_index(drop=True)
            br = sr.head(1)
            best_report.append(br.copy())
            sorted_report.append(sr.copy())
        self.best_report = pd.concat(best_report,0).reset_index(drop=True)
        self.sorted_report = pd.concat(sorted_report,0).reset_index(drop=True)
        print("Report ready. Call obj.best_report or obj.sorted_report")
    def get_predictor(self):
        best_report = self.best_report.copy(deep=True) 
        best_report['params_str'] = best_report[['params','regressor']].apply(lambda x: x[1]+"_"+str(x[0]),1)
        uregs = best_report.sort_values(by=['params_str'])['params_str'].unique()
        all_preds = []
        for ureg in uregs:
            dfr = best_report[best_report['params_str']==ureg]
            popular_reg = dfr.copy()
            reg_name = popular_reg['regressor'].unique()[0]
            reg_params = popular_reg['params'].head(1).values[0]
            reg_var_names = popular_reg['Variable'].values
            reg_var_index = list(popular_reg.index)
            #print(reg_name,reg_params,reg_var_index)
            regx = self.available_regressors[reg_name](**reg_params)
            regx.fit(self.x_train,self.y_train)
            pred_test = regx.predict(self.x_test)
            coefs = regx.coef_.transpose().copy()
            intercept = regx.intercept_.copy()
            all_preds.append({'pred':pred_test.copy(),'reg_name':reg_name,'coef':coefs.copy(),'intercept':intercept.copy(),
                     'params':reg_params,'var_index':reg_var_index[:],'var_names':reg_var_names[:]})
        final_coef = np.zeros((self.y.shape[1],self.X.shape[1]))
        final_int = np.zeros((self.y.shape[1],self.X.shape[1]))
        for regi in range(len(all_preds)):
            final_coef[all_preds[regi]['var_index'],:] = all_preds[regi]['coef'].transpose()[all_preds[regi]['var_index'],:]
            _inter = all_preds[regi]['intercept'].reshape(all_preds[regi]['intercept'].shape[0],1)
            final_int[all_preds[regi]['var_index'],:] = _inter[all_preds[regi]['var_index'],:]
        self.final_coef = final_coef.copy()
        self.final_intercept = final_int.copy()
    def get_train_test(self):
        test_split_int = int(round(self.X.shape[0]*self.test_split))
        all_index = list(np.arange(self.X.shape[0]))
        test_index = np.random.choice(all_index,test_split_int,replace=False)
        train_index = list(filter(lambda x : x not in test_index, all_index))
        self.x_train, self.x_test = self.X[train_index,:],self.X[test_index,:]
        self.y_train, self.y_test = self.y[train_index,:],self.y[test_index,:]
    def predict(self,input_pred):
        return np.dot(input_pred,self.final_coef.transpose())+self.final_intercept[:,0]
    def get_test_pred_diff(self,norm=True):
        ynt, ynts, ynpts = norm_all(self.y_train,self.y_test,self.predict(self.x_test))
        self.test_pred_diff_norm = np.abs(ynts-ynpts)
        self.test_pred_diff_norm_mean = self.test_pred_diff_norm.mean(0)
        self.test_pred_diff_norm_std = self.test_pred_diff_norm.std(0)
        self.test_pred_diff_norm_mean_std_up = self.test_pred_diff_norm_mean + self.test_pred_diff_norm_std
        self.test_pred_diff_norm_mean_std_down = self.test_pred_diff_norm_mean - self.test_pred_diff_norm_std
        print_txt = "Variables generated: self.test_pred_diff_norm,\nself.test_pred_diff_norm_mean\n"
        print_txt+="self.test_pred_diff_norm_std\n"
        print_txt+="self.test_pred_diff_norm_mean_std_up\n"
        print_txt+="self.test_pred_diff_norm_mean_std_down\n"
        print(print_txt)
    def get_test_train_diff(self,norm=True):
        sample_train_index = np.random.choice(np.arange(self.y_train.shape[0]),self.y_test.shape[0],replace=False)
        ynt, ynts, ynpts = norm_all(self.y_train,self.y_test,self.predict(self.x_test))
        self.test_train_diff_norm = np.abs(ynts-ynt[sample_train_index,:])
        self.test_train_diff_norm_mean = self.test_train_diff_norm.mean(0)
        self.test_train_diff_norm_std = self.test_train_diff_norm.std(0)
        self.test_train_diff_norm_mean_std_up = self.test_train_diff_norm_mean + self.test_train_diff_norm_std
        self.test_train_diff_norm_mean_std_down = self.test_train_diff_norm_mean - self.test_train_diff_norm_std
        print_txt = "Variables generated: self.test_train_diff_norm,\nself.test_train_diff_norm_mean\n"
        print_txt+="self.test_train_diff_norm_std\n"
        print_txt+="self.test_train_diff_norm_mean_std_up\n"
        print_txt+="self.test_train_diff_norm_mean_std_down\n"
        print(print_txt)








