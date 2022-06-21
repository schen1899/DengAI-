import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import PreProcess as PP


if __name__ == "__main__":
    prep = False
    modeloptim = True
    submit = True

    if prep == True:
        #Pre-Processing w/ Rm of Nan
        data = PP.Pre_Processing(mode='KNN')
        cov = data.cov_matrix()
        rm_list = data.mk_rm_list(cov)
        data.rm_data()
        data.Z_normalize()
        y = data.train_label
        x = data.final_features
        #np.savetxt("x_KNN.csv", x,delimiter=',')
        #np.savetxt("y_KNN.csv", y, delimiter=',')

        if submit == True:
            # create model
            x = pd.read_csv("x_KNN.csv", header=None).to_numpy()
            y = np.ravel(pd.read_csv("y_KNN.csv", header=None).to_numpy())
            regr = RandomForestRegressor(random_state=0, min_samples_leaf=100, n_estimators=500)
            regr.fit(x, y)
            # Do Pre-processing on X_Test Data
            data_test = PP.Pre_Processing(data='test', model=data.knn_models)
            data_test.final_features = np.delete(data_test.features, rm_list, 1)
            data_test.Z_normalize()
            feature_test = data_test.final_features
            y_test = regr.predict(feature_test).round().astype(np.int64)
            submitdf = pd.read_csv('submission_format.csv')
            submitdf.iloc[:, -1] = y_test
            submitdf.to_csv('y_min_samples_leaf100_n_estimator_500.csv', index=False)
            # np.savetxt("y_test.csv", y_test, delimiter=',')

        else:
            pass


    elif modeloptim == True:
        #optimizing model with train data
        x = pd.read_csv("x_KNN.csv",header=None).to_numpy()
        y = np.ravel(pd.read_csv("y_KNN.csv", header=None).to_numpy())
        random_state = 0
        RFR = RandomForestRegressor(random_state=random_state)
        #Model parameters
        param = {
            'n_estimators': [300,400,500,700],
            'min_samples_leaf':[10,15,20,30,50,70,100]
        }
        GS = GridSearchCV(estimator=RFR,
                          param_grid=param,
                          scoring=['max_error',"neg_root_mean_squared_error"],
                          refit="neg_root_mean_squared_error",
                          n_jobs=-1
                          )
        GS.fit(x,y)
        print(GS.best_estimator_)
        df = pd.DataFrame(GS.cv_results_)
        df.to_csv('CV_RandomForest3.csv')