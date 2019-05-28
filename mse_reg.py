from sklearn.linear_model import LinearRegression,Ridge,Lasso,MultiTaskLasso,ElasticNet,MultiTaskElasticNet,LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge,ARDRegression,SGDRegressor,PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor,TheilSenRegressor,HuberRegressor
import scipy
from skopt import gp_minimize
import skopt
from skopt.space import Real, Integer, Categorical
import numpy as np
class FeatureRegressor:
    def __init__(self,ix,iy,regressors,target_labels=[],
        test_split = 0.25, repeat=5,train_weight=1,test_weight=2,skopt_kargs={'n_calls':10},v=False):
        """
            regressors: a dictionary with the form:
            regressors = {'linear_regression':{},
                            'ridge':{'init_params':{'random_state':0},
                                    'opt_params':{'alpha':[0.01,1.0]}
                                    }
                            }
        """
        self.available_regressors = {'linear_regression':LinearRegression,'ridge':Ridge,'lasso':Lasso,
        'multi_task_lasso':MultiTaskLasso,'elastic_net':ElasticNet,'multi_task_elastic_net':MultiTaskElasticNet,
        'lasso_lars':LassoLars,'orthogonal_matching_pursuit':OrthogonalMatchingPursuit,
        'bayesian_ridge':BayesianRidge,'ard':ARDRegression,'sgd':SGDRegressor,'passive_aggressive':PassiveAggressiveRegressor,
        'ransac':RANSACRegressor,'theil_sen':TheilSenRegressor,'huber':HuberRegressor}
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
        self.best_report = {}
        self.optimizer = None
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
    
    def cross_val_val(self):
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
        return mse_all
    
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
        for name,value in zip(self.current_opt_params_names,opt_params):
            self.current_regressor_params[name]=value
        #Validate if params exist in current regressor
        if self.v ==True:
            if type(self.current_target_label)==str:
                target_label = self.current_target_label
            print(self.current_regressor_name,"Initialized for:",target_label)
        return self.cross_val_val()

    def optimize_regressor(self):
        from skopt import gp_minimize
        best_name_value = {}
        if len(self.current_opt_params_values)>0:
            self.optimizer = gp_minimize(self.cross_val_rep,self.current_opt_params_values,**self.skopt_kargs)
            for name, value in zip(self.current_opt_params_names,self.optimizer.x):
                best_name_value[name]=value
            value_function = self.optimizer.fun
        else:
            value_function = self.cross_val_val()
        self.report[self.current_target_label][self.current_regressor_name]={'best_value_function':value_function,
                                                    'best_opt_params':best_name_value.copy(),
                                                    'init_params':self.current_regressor_init_params.copy()}
        if self.v==True:
            print(self.report)
    def start_optimization(self):
        for feature_col in range(self.total_features):
            self.current_target_index = feature_col
            if len(self.y.shape)==1:
                self.y = self.y.reshape(self.y.shape[0],1)
            self.yi = self.y[:,feature_col:feature_col+1]
            if len(self.yi.shape)==1:
                self.yi = self.yi.reshape(self.yi.shape[0],1)
            if len(self.target_labels)!=self.total_features:
                self.current_target_label = "var_"+str(self.current_target_index)
            else:
                self.current_target_label = self.target_labels[self.current_target_index]
            self.report[self.current_target_label]={}
            self.best_report[self.current_target_label]={}
            for regressor_name in self.regressors.keys():
                self.current_regressor_name=regressor_name
                self.init_regressor()
                self.optimize_regressor()
            self.best_report[self.current_target_label]['value'] = 1e100
            self.best_report[self.current_target_label]['name'] = ""
            self.best_report[self.current_target_label]['params'] = {}
            print("Starting:",self.current_target_label,feature_col,"/",self.total_features)
            for regx in self.regressors.keys():
                _ = self.report[self.current_target_label][regx]
                reg_val = _['best_value_function']
                reg_opt = _['best_opt_params']
                reg_init = _['init_params']
                if  reg_val < self.best_report[self.current_target_label]['value']:
                    self.best_report[self.current_target_label]['value']=reg_val
                    self.best_report[self.current_target_label]['name']=regx
                    self.best_report[self.current_target_label]['params']=reg_opt.copy()
                    self.best_report[self.current_target_label]['params'].update(reg_init.copy())
            print("Finished:",self.current_target_label,feature_col,"/",self.total_features)
