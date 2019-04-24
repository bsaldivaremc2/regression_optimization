# regression_optimization  
A library to find the best hyperparameters using Gaussian optimization for scikit-learn regressors. 

It will evaluate the target "y" for each feature individually.  
The description will be added soon. In summary, it uses a t test to see if the regression differences between  
the ground truth and the predicted in the test test doesn't have significant differences.  
It uses a cross validation with reposition.

Just define X and y with the shape [number of samples,number of features]

#How to use:  
```python
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

### Try multithreading with multiple objects  
```python
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
```
