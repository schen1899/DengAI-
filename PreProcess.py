import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
from sklearn.neighbors import KNeighborsRegressor

class Pre_Processing(object):

    def __init__(self, mode='KNN', data='train', model={}):
        """
        :param mode: this determines how the data is processed for Nans
        rm means to remove the rows
        mean = set to avg
        mode = set to mode
        perdict = to perdict
        """
        if mode == 'KNN':
            if data == 'train':
                self.knn_models = {}
            elif data == 'test':
                self.knn_models = model

        #rm Nan based on mode
        self.rm_NaN(mode=mode,data=data)

        self.dict = {}
        self.rm_feature_list = []
        if data == "train":
            self.train_label = pd.read_csv('dengue_labels_train.csv').iloc[:,-1].to_numpy()


    def rm_NaN(self,mode='mean', data='train'):
        """
        :param mode: the mode will determine the Nan replacement that occurs from cols 3: on the features  array.
        Several inputs as Mean, KNN, Min, Max, Median can be used
        :return: a features array of non-nans processed via a mode (mean, min, max, knn)
        """
        if data == 'train':
            #read features into a df
            feature_w_nan = pd.read_csv('dengue_features_train.csv')
            ft_arr_w_nan = feature_w_nan.to_numpy()

            # Read labels into df
            label = pd.read_csv('dengue_labels_train.csv')

            # Ensure consistency with the city, year, weekofyear to match up label and features
            feature_checkdf = feature_w_nan.iloc[:, 0:3]
            label_checkdf = label.iloc[:, 0:3]
            bool_checkdf = feature_checkdf != label_checkdf
            tuple_check = np.nonzero(bool_checkdf.to_numpy())
            if tuple_check[0].size != 0:
                print('failed consistency between label and feature rows')
                sys.exit(1)
            else:
                pass
        else:
            feature_w_nan = pd.read_csv('dengue_features_test.csv')
            ft_arr_w_nan = feature_w_nan.to_numpy()

            # Read labels into df
            label = pd.read_csv('submission_format.csv')

            # Ensure consistency with the city, year, weekofyear to match up label and features
            feature_checkdf = feature_w_nan.iloc[:, 0:3]
            label_checkdf = label.iloc[:, 0:3]
            bool_checkdf = feature_checkdf != label_checkdf
            tuple_check = np.nonzero(bool_checkdf.to_numpy())
            if tuple_check[0].size != 0:
                print('failed consistency between label and feature rows')
                sys.exit(1)
            else:
                pass

        self.time = feature_w_nan.iloc[:, 0:4]
        #Pre-process array to remove strings in the feature t arrray (column 0 to 3)
        str_col = [0,1,2,3]

        ft_arr_w_nan = np.delete(ft_arr_w_nan, str_col, axis=1).astype(float)
        nan_index = np.argwhere(np.isnan(ft_arr_w_nan))
        #nan index in tuple form:
        nan_tuples = np.nonzero(np.isnan(ft_arr_w_nan))
        #Get unique nans
        nan_col = np.unique(nan_index[:,1])


        ft_arr_calc = np.copy(ft_arr_w_nan)

        #Pre-process the index to remove colummns stated in str_col for tracking
        self.features_index = feature_w_nan.columns.delete(str_col).to_numpy()

        #label_df to np array
        label_np = label.iloc[:, -1].to_numpy()

        #logic to seperate nan vs non-nan data by each nan_column:
        #logic to calculate values for each column with nan and replace the [row,col] nan  with the specified method
        for col_index in nan_col:
            # calculate mode
            if mode == "mean":
                #calculate mean
                col_calc_value = np.nanmean(ft_arr_w_nan[:,col_index])
            elif mode == "min":
                #calculate min
                pass
            elif mode == "max":
                #calculate max
                pass
            elif mode == "median":
                #calculate median
                pass
            elif mode == "KNN":
                if data == 'train':
                    #calculate column without nans to feed into KNN
                    # logic to seperate nan vs non-nan data by each nan_column:
                    nan_tuple_col_index = np.nonzero(nan_tuples[1] == col_index)
                    nan_rows = nan_tuples[0][nan_tuple_col_index]
                    #np pop nan_rows ft_array_w_nan[col_index].rm(nan_rows), labels.rm(nan_rows)
                    ft_KNN = np.delete(ft_arr_w_nan[:, col_index], nan_rows)
                    label_KNN = np.delete(label_np, nan_rows)
                    #create KNN Model
                    #default neighbor = 5
                    KNN_Model = KNeighborsRegressor(n_neighbors=2)
                    KNN_Model.fit(np.expand_dims(label_KNN, axis=1), ft_KNN)
                    self.knn_models[col_index] = KNN_Model
                else:
                    pass
            else:
                print('Mode is not understood by rm_nan')
                sys.exit(1)


            #col_calc.append([col_index, col_calc_value]) works for mean right not. nm
            row_index = np.argwhere(np.isnan(ft_arr_w_nan[:,col_index])).flatten()
            for row in row_index:
                if mode == 'mean':
                    ft_arr_calc[row, col_index] = col_calc_value
                elif mode == "KNN":
                    #Query  KNN model with X = Ft_KNN, Y = label_KNN
                    ft_arr_calc[row, col_index] = self.knn_models[col_index].predict(label_np[row].reshape(-1,1))



        #Assign the final features array to self.features
        self.features = ft_arr_calc

        return

    def init_dict(self):
        """
        This function will create an intialized dict of {descriptor indexes : 0 count} for copying
        :return:
        """
        # intialize dict {descriptor : counts}
        for descriptors in range(len(self.features_index)):
            self.dict[descriptors] = 0
        return

    def cov_matrix(self, check=True):
        """
        :return:  the co-variance matrix np array with shape (features x features) containing
        """
        self.cov = np.corrcoef(self.features, rowvar=False)

        #Check CSV
        if check == True:
            df = pd.DataFrame(self.cov,columns=self.features_index)
            df.to_csv('cov.csv',index=False)

        #intialize dict {descriptor : counts}
        self.init_dict()
        return self.cov

    def mk_rm_list(self,cov, lim=0.6):
        neg_corr = np.where(cov <= -lim)
        corr = np.where(cov >= lim)
        temp_dict = self.count_corr(corr, neg_corr)
        keys = np.fromiter(temp_dict.keys(), dtype=np.int)
        values = np.fromiter(temp_dict.values(), dtype=np.int)
        if np.where(values > 1)[0].shape[0] == 0:
            self.rm_feature_list = np.sort(self.rm_feature_list)
            return self.rm_feature_list
        else:
            #remove highest count in the future
            key_max = np.where(values == values.max())[0]
            if key_max.shape[0] != 1:
                #key_max = np.random.choice(key_max)
                key_max = key_max[0].item()
            else:
                key_max = key_max.item()
            self.rm_feature_list.append(key_max)
            cov[key_max,:] = 0
            cov[:,key_max] = 0
            return self.mk_rm_list(cov,lim)

    def count_corr(self,corr,neg_corr):
        """
        :param corr: correlation above lim set in function rm_feature
        :param neg_corr: negative correlation below lim set in function rm_feature
        :return: the temporary dict housing {descriptor index : # of times correlated with other descriptors}
        """
        temp_dict = self.dict.copy()
        #counts for the positive correlation
        for element in corr[0]:
            temp_dict[element] = temp_dict[element] + 1
        for element in neg_corr[0]:
            temp_dict[element] = temp_dict[element] + 1
        return temp_dict

    def rm_data(self):
        self.final_features = np.delete(self.features, self.rm_feature_list, 1)
        self.final_features_index = np.delete(self.features_index, self.rm_feature_list)
        return

    def Z_normalize(self):
        """
        Perform Z Normalization on all the final_feature descriptors
        :return:
        """
        for descriptors in range(self.final_features.shape[1]):
            mean = self.final_features[:, descriptors].mean()
            std_dev = self.final_features[:, descriptors].std()
            self.final_features[:, descriptors] = (self.final_features[:, descriptors] - mean)/std_dev
        return

    def mk_histogram(self):
        """
        generates histrogram plots for features data set as png
        :return:
        """
        num_bins = round(self.features.shape[0] ** 0.5)
        #UL = self.features.max(axis=0)
        #LL = self.features.min(axis=0)
        #bin_width = (UL - LL) / num_bins
        for ft in range(self.features.shape[1]):
            plt.hist(self.features[:,ft], bins=num_bins)
            plt.xlabel(self.features_index[ft])
            plt.ylabel('counts')
            plt.title(self.features_index[ft])
            plt.savefig(self.features_index[ft] + '.png')
            plt.close()
