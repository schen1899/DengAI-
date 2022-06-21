import pandas as pd
from prophet.diagnostics import cross_validation
import prophet as Prophet
import numpy as np
import itertools
from prophet.diagnostics import performance_metrics
import logging
import sys



def tseries_df(data, type):
    """
    :param data: raw dataframe including columns, city, week_start_date, and total_cases
    :return:
    """
    if type == 'train':
        df_times = data[['city','week_start_date', 'total_cases']].copy()
        df_times.set_index('city', inplace=True)
        df_times.rename(columns={'week_start_date':'ds', 'total_cases':'y'}, inplace=True)
    else:
        df_times = data[['city','week_start_date']].copy()
        df_times.set_index('city', inplace=True)
        df_times.rename(columns={'week_start_date':'ds'}, inplace=True)

    return df_times


if __name__ == "__main__":
    x = pd.read_csv('dengue_features_train.csv')
    y = pd.read_csv('dengue_labels_train.csv')
    y_test = pd.read_csv('dengue_features_test.csv')
    #create charts for the dataset
    cols = list(x.columns)
    meta_x = cols[:3]
    ft_x = cols[3:]
    data = x.join(y.loc[:,'total_cases'])
    data = data.ffill(axis=0)

    #toggle to turn scritpt to perform cross validation on train set
    cross_val = False

    #perform pre-processing to use prophet
    df_times = tseries_df(data,type='train')
    #pre-process test data to be able to generate forecasting data
    df_test = tseries_df(y_test, type='test')

    #intialize dicts for storing models per city & training dates
    cities_dict = {}
    model_dict = {}
    cities_list = ['sj', 'iq']

    #CrossValidation Required to tune flexibility and seasonality; best model parameters are placed here
    city_param_grid = {
        'sj' : {
            'changepoint_prior_scale': 0.01,
            'seasonality_prior_scale': 0.1
        },
        'iq' : {
            'changepoint_prior_scale': 0.001,
            'seasonality_prior_scale': 1
        }

    }

    #cross validation grid to search parameters
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 0.5, 1.0, 10.0],
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = {}  # Store the RMSEs for each params here
    tuning_results = {} #store tuning results

    #make Prophet disable logs
    logging.getLogger('fbprophet').setLevel(logging.WARNING)

    #Perform Cross Validation, Training, Forecasting for the cities within the dataframe
    for city in cities_list:
        cities_dict[city] = {}
        model_dict[city] = {}
        rmses[city] = []

        #df slice in city for usage
        df_slice = df_times.iloc[df_times.index == city].copy()

        #Cross-Validation to update city_param_grid[city]
        if cross_val == True:
            #Cross-Validation Section

            # Use cross validation to evaluate all parameters
            for params in all_params:
                m = Prophet.Prophet(**params).fit(df_slice)  # Fit model with given params
                df_cv = cross_validation(m, period='90 days', horizon = '365 days', parallel="processes")
                df_p = performance_metrics(df_cv)
                rmses[city].append(df_p['rmse'].values[0])

            # Find the best parameters
            tuning_results[city] = pd.DataFrame(all_params)
            tuning_results[city]['rmse'] = rmses[city]
            tuning_results[city].to_csv('tuning_rmses_{}.csv'.format(city))

            #set hyperparameters from rmses minimum
            city_param_grid[city] = all_params[np.argmin(rmses[city])]

            # pre-process train data to perform cross-validation for
            df_test = tseries_df(data, type='train')


        # Training Section
        cities_dict[city]['train'] = df_slice
        model_dict[city]['model'] = Prophet.Prophet(changepoint_prior_scale=city_param_grid[city]['changepoint_prior_scale'],
                                                    seasonality_prior_scale=city_param_grid[city]['seasonality_prior_scale']).fit(df_slice)

        #Forcasting section on Test Data
        df_slice = df_test.iloc[df_test.index == city].copy()
        cities_dict[city]['test'] = df_slice
        model_dict[city]['forecast_test'] = model_dict[city]['model'].predict(df_slice)
        model_dict[city]['forecast_test'].insert(0,'city',city)
        fig1 = model_dict[city]['model'].plot(model_dict[city]['forecast_test'])
        fig1.savefig('{}_{}_{}_Test.png'.format(city,city_param_grid[city]['changepoint_prior_scale'],
                                                city_param_grid[city]['seasonality_prior_scale']))

        #Forcasting section on Train Data
        df_slice = df_times.iloc[df_times.index == city].copy()
        cities_dict[city]['test'] = df_slice
        model_dict[city]['forecast_train'] = model_dict[city]['model'].predict(df_slice)
        model_dict[city]['forecast_train'].insert(0,'city',city)
        fig1 = model_dict[city]['model'].plot(model_dict[city]['forecast_train'])
        fig1.savefig('{}_{}_{}_Train.png'.format(city,city_param_grid[city]['changepoint_prior_scale'],
                                                city_param_grid[city]['seasonality_prior_scale']))

    if cross_val == True:
        sys.exit(1)

    submission_df = pd.concat([model_dict['sj']['forecast_test'], model_dict['iq']['forecast_test']])

    #make calculations to yhat to preict higher than avg or not using a supervised model.

    submission_df = submission_df[['city', 'ds', 'yhat']].copy()
    submission_df.rename(columns={'ds':'week_start_date', 'yhat':'total_cases'}, inplace=True)
    submission_df.loc[:,'total_cases'] = submission_df.loc[:,'total_cases'].astype(np.int64)

    #make y_test week_start_date to datatime dtype
    y_test[['week_start_date']] = pd.DataFrame(pd.to_datetime(y_test[['week_start_date']].squeeze()))

    submission_df.set_index(['city','week_start_date'], inplace=True)
    y_test.set_index(['city','week_start_date'], inplace=True)
    #the index contais 000:00:00 making it not possible to join on index
    submission_df = submission_df.join(y_test[['year','weekofyear']])
    submission_df.reset_index(inplace=True)
    submission_df.drop(['week_start_date'], axis=1, inplace=True)

    #move columns indexes around
    new_cols = ['city','year','weekofyear','total_cases']
    submission_df = submission_df.reindex(columns=new_cols)

    submission_df.to_csv('Prophet.csv'.format(), index=False)