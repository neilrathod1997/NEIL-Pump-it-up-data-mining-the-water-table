# importing necessary libraries


from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.metrics import  balanced_accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.ensemble import VotingClassifier
from joblib import dump, load
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.base import BaseEstimator, TransformerMixin


class My_LabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, df, df_y):
        maps_ = {}
        for col in df:
            y = df[col]
            uni = np.unique(y)
            map_ = {}
            for c in uni:
                map_[c] = len(map_)
            maps_[col] = map_
        self.maps_ = maps_
        return self

    def transform(self, df):
        ndf = df.copy()
        for col in df:
            ny = []
            map_ = self.maps_[col]
            for c in np.array(df[col]):
                if c in self.maps_[col]:
                    ny.append(self.maps_[col][c])
                else:
                    ny.append(-1)
            ndf[col] = ny
        return ndf


# import data
train_labels = pd.read_csv('training_set_labels.csv')
train_values = pd.read_csv('training_Set_values.csv', parse_dates=['date_recorded'], na_values=[0, '0'])
test_values = pd.read_csv('test_set_value.csv', parse_dates=['date_recorded'], na_values=[0, '0'])

# merge train values and train labels to single dataset
train = pd.merge(train_labels, train_values, on='id')
test = test_values.copy()


def final_fun_2(X):
    """ THIS FUNCTION TAKES RAW DATA AND DOSE PREPROCESSING AND FEATURE ENGG. AND predict using best model"""
    test = X.copy()
    # longitude
    means_longitude_subvillage = pd.read_csv('means_longitude_subvillage.csv')
    means_longitude_ward = pd.read_csv('means_longitude_ward.csv', )
    means_longitude_lga = pd.read_csv('means_longitude_lga.csv', )
    means_longitude_region = pd.read_csv('means_longitude_region.csv', )

    # merge the aggregated dataframes as new columns to the original df this will make it easier to replace missing values
    test = test.merge(means_longitude_subvillage, how='left', on=['region', 'lga', 'ward', 'subvillage'])
    test = test.merge(means_longitude_ward, how='left', on=['region', 'lga', 'ward'])
    test = test.merge(means_longitude_lga, how='left', on=['region', 'lga'])
    test = test.merge(means_longitude_region, how='left', on=['region'])

    # select the right longitude level based on the availability of information
    test['imputed_longitude'] = np.where(test['longitude'].isna(), test['longitude_imputed_subvillage'], test[
        'longitude'])  # if longitude is missing, impute it by the mean of the subvillage
    test['imputed_longitude'] = np.where(test['imputed_longitude'].isna(), test['longitude_imputed_ward'], test[
        'imputed_longitude'])  # if subvillage mean is missing, impute it by the ward
    test['imputed_longitude'] = np.where(test['imputed_longitude'].isna(), test['longitude_imputed_lga'],
                                         test['imputed_longitude'])
    test['imputed_longitude'] = np.where(test['imputed_longitude'].isna(), test['longitude_imputed_region'],
                                         test['imputed_longitude'])

    # drop redundant columns
    test = test.drop(
        ['longitude_imputed_subvillage', 'longitude_imputed_ward', 'longitude_imputed_lga', 'longitude_imputed_region',
         'longitude'], axis=1)

    # Premit
    permit_mg_mode = pd.read_csv('permit_mg_mode3.csv')

    permit_mg_mode = permit_mg_mode.rename(columns={"permit": "imputed_permit_mg"})
    test = test.merge(permit_mg_mode, how='left', on=['public_meeting', 'management_group'])

    test['imputed_permit'] = np.where(test['permit'].isna(), test['imputed_permit_mg'], test[
        'permit'])  # if permit is missing, replace it by the mode of public meeting - management group
    test['imputed_permit'] = np.where(test['imputed_permit'].isna(), test['permit'].mode(), test[
        'imputed_permit'])  # if eitther public meeting or management group is missing, then use the mode of permit (True)

    # drop original permit column
    test = test.drop(['permit', 'imputed_permit_mg'], axis=1)

    #  Public Meeting
    # True is  mode of public meeting.
    # Over 90% of the pumps have a public meeting. I will therefore impute by the mode.
    test['public_meeting'] = test['public_meeting'].fillna(True)

    # Scheme management
    scheme_mode = pd.read_csv('permit_mg_mode.csv')

    # merge scheme_mode to original df and use it to replace missing values
    test = test.merge(scheme_mode, how='left', on=['management'])
    test['imputed_scheme__management'] = np.where(test['scheme_management'].isna(), test['imputed_scheme_management'],
                                                  test['scheme_management'])

    # drop redundant columns
    test = test.drop(['scheme_management', 'imputed_scheme_management'], axis=1)

    # Installer
    test['installer'] = test['installer'].str.lower()

    # plot top 10 installers
    test['installer'] = np.where(test['installer'] == 'gove', 'gover', test['installer'])
    test['installer'] = np.where(test['installer'] == 'community', 'commu', test['installer'])
    test['installer'] = np.where(test['installer'] == 'danid', 'danida', test['installer'])

    inst150 = pd.read_csv('inst150.csv')
    top_installers = np.array(inst150['0'])
    # replace funders that are not in top 10 with 'other'
    test['installer'] = np.where(test['installer'].isin(top_installers), test['installer'], 'other')

    # Funder
    # set al entries to lowercase
    test['funder'] = test['funder'].str.lower()

    fund150 = pd.read_csv('fundt150.csv')
    top_funders = np.array(fund150['0'])

    # replace funders that are not in top 150 with 'other'
    test['funder'] = np.where(test['funder'].isin(top_funders), test['funder'], 'other')

    # Construction Year
    mean_construction = pd.read_csv('mean_construction.csv')

    mean_construction = mean_construction.rename(columns={"construction_year": "imputed_construction_year"})

    # merge this df to the main df and replace missing values
    test = test.merge(mean_construction, how='left', on='extraction_type_group')
    test['construction_year_imputed'] = np.where(test['construction_year'].isna(), test['imputed_construction_year'],
                                                 test['construction_year'])

    # drop redundant columns
    test = test.drop(['imputed_construction_year', 'construction_year'], axis=1)

    # GPS height
    # subvillage
    means_altitude_subvillage = pd.read_csv('means_altitude_subvillage.csv', )

    # ward level
    means_altitude_ward = pd.read_csv('means_altitude_ward.csv')

    # lga level
    means_altitude_lga = pd.read_csv('means_altitude_lga.csv')

    # region level
    means_altitude_region = pd.read_csv('means_altitude_region.csv')

    # region basin
    means_altitude_basin = pd.read_csv('means_altitude_basin.csv')

    # merge the aggregated dataframes as new columns to the original df
    test = test.merge(means_altitude_subvillage, how='left', on=['region', 'lga', 'ward', 'subvillage'])
    test = test.merge(means_altitude_ward, how='left', on=['region', 'lga', 'ward'])
    test = test.merge(means_altitude_lga, how='left', on=['region', 'lga'])
    test = test.merge(means_altitude_region, how='left', on=['region'])
    test = test.merge(means_altitude_basin, how='left', on=['basin'])

    # create final imputed longitude column
    test['imputed_gps_height'] = np.where(test['gps_height'].isna(), test['gps_height_imputed_subvillage'], test[
        'gps_height'])  # if longitude is missing, impute it by the mean of the subvillage
    test['imputed_gps_height'] = np.where(test['imputed_gps_height'].isna(), test['gps_height_imputed_ward'], test[
        'imputed_gps_height'])  # if subvillage mean is missing, impute it by the ward
    test['imputed_gps_height'] = np.where(test['imputed_gps_height'].isna(), test['gps_height_imputed_lga'],
                                          test['imputed_gps_height'])
    test['imputed_gps_height'] = np.where(test['imputed_gps_height'].isna(), test['gps_height_imputed_region'],
                                          test['imputed_gps_height'])
    test['imputed_gps_height'] = np.where(test['imputed_gps_height'].isna(), test['gps_height_imputed_basin'],
                                          test['imputed_gps_height'])

    # drop redundant columns
    test = test.drop(['gps_height_imputed_subvillage', 'gps_height_imputed_ward', 'gps_height_imputed_lga',
                      'gps_height_imputed_region', 'gps_height', 'gps_height_imputed_basin'], axis=1)

    # Population

    # subvillage
    means_population_subvillage = pd.read_csv('means_population_subvillage.csv')

    # ward level
    means_population_ward = pd.read_csv('means_population_ward.csv')

    # lga level
    means_population_lga = pd.read_csv('means_population_lga.csv')

    # region level
    means_population_region = pd.read_csv('means_population_region.csv')

    # region basin
    means_population_basin = pd.read_csv('means_population_basin.csv')

    # merge the aggregated dataframes as new columns to the original df
    test = test.merge(means_population_subvillage, how='left', on=['region', 'lga', 'ward', 'subvillage'])
    test = test.merge(means_population_ward, how='left', on=['region', 'lga', 'ward'])
    test = test.merge(means_population_lga, how='left', on=['region', 'lga'])
    test = test.merge(means_population_region, how='left', on=['region'])
    test = test.merge(means_population_basin, how='left', on=['basin'])

    # create final imputed longitude column
    test['imputed_population'] = np.where(test['population'].isna(), test['population_imputed_subvillage'], test[
        'population'])  # if longitude is missing, impute it by the mean of the subvillage
    test['imputed_population'] = np.where(test['imputed_population'].isna(), test['population_imputed_ward'], test[
        'imputed_population'])  # if subvillage mean is missing, impute it by the ward
    test['imputed_population'] = np.where(test['imputed_population'].isna(), test['population_imputed_lga'],
                                          test['imputed_population'])
    test['imputed_population'] = np.where(test['imputed_population'].isna(), test['population_imputed_region'],
                                          test['imputed_population'])
    test['imputed_population'] = np.where(test['imputed_population'].isna(), test['population_imputed_basin'],
                                          test['imputed_population'])

    # drop redundant columns
    test = test.drop(['population_imputed_subvillage', 'population_imputed_ward', 'population_imputed_lga',
                      'population_imputed_region', 'population', 'population_imputed_basin'], axis=1)

    # change type to categorical

    test['num_private'] = test['num_private'].astype('str')
    test['region_code'] = test['region_code'].astype('str')
    test['district_code'] = test['district_code'].astype('str')
    test['num_private'] = test['num_private'].astype('str')

    # replace string to integer
    test['public_meeting'] = test['public_meeting'].replace({True: 1, False: 0})
    test['imputed_permit'] = test['imputed_permit'].replace({True: 1, False: 0})

    # change to integer
    test[['imputed_gps_height', 'construction_year_imputed', 'imputed_population']] = test[
        ['imputed_gps_height', 'construction_year_imputed', 'imputed_population']].astype('int')

    # change type to categorical

    # remove decimal
    test['district_code'] = test['district_code'].str.split(".").str[0]

    test = test.rename(columns={"imputed_permit": "permit", "imputed_scheme__management": "scheme_management",
                                "imputed_gps_height": "gps_height", 'construction_year_imputed': 'construction_year',
                                'imputed_population': 'population', 'imputed_longitude': 'longitude'}, errors="raise")

    final_df_test = test.copy()

    # create age feature
    final_df_test['recorded_year'] = pd.DatetimeIndex(final_df_test['date_recorded']).year
    final_df_test['age'] = final_df_test['recorded_year'] - final_df_test['construction_year']
    final_df_test = final_df_test.drop('recorded_year', axis=1)

    # Season

    final_df_test['month'] = pd.DatetimeIndex(final_df_test['date_recorded']).month

    # season encoder based on reports by water aid tanzania and https://tanzania-specialist.com/best-time-to-visit-tanzania/
    season_mapper = {1: 'short dry', 2: 'short dry', 3: 'long rain', 4: 'long rain', 5: 'long rain', 6: 'long dry',
                     7: 'long dry', 8: 'long dry', 9: 'long dry', 10: 'long dry', 11: 'short rain', 12: 'short rain'}
    # .p feature values to scale
    final_df_test['season'] = final_df_test['month'].replace(season_mapper)
    final_df_test = final_df_test.drop('month', axis=1)

    # Amount tsh missing

    # where amount tsh isn't missing, the percentage of functional pumps is a lot higher
    final_df_test['amount_tsh_missing'] = np.where(final_df_test['amount_tsh'].isna(), 1, 0)

    # Region District
    final_df_test['region_district'] = final_df_test['region'] + "-" + final_df_test['district_code']

    # two decimal places is 1.1 km accurate. This will provide enough information on the location. Using the full coordinate doesn't provide a lot of general information, but does result in high cardinality
    final_df_test['longitude'] = round(final_df_test['longitude'], 2)
    final_df_test['latitude'] = round(final_df_test['latitude'], 2)

    # i want to keep extraction type class and I will group the extraction type group en type together

    # swn 80 and swn 81 become swn
    # cemo + climax become other motorpump
    # other -mkulima, other -play and walimi become other handpump

    swn = ['other - swn 81', 'swn80']
    final_df_test['extraction_type'] = np.where(final_df_test['extraction_type'].isin(swn), 'swn',
                                                final_df_test['extraction_type'])

    other_handpump = ['other - mkulima/shinyanga', 'other - play pump', 'other - walimi']
    final_df_test['extraction_type'] = np.where(final_df_test['extraction_type'].isin(other_handpump), 'other handpump',
                                                final_df_test['extraction_type'])

    other_motorpump = ['cemo', 'climax']
    final_df_test['extraction_type'] = np.where(final_df_test['extraction_type'].isin(other_motorpump),
                                                'other motorpump', final_df_test['extraction_type'])

    # based on reports by water aid
    # non autonomous = government, VWC, town council ..... also water authority, parastatal (=state company) SWC
    # autonomous = WUA, WUG, board, trust, school
    # private = private, company

    non = ['VWC', 'Water authority', 'Parastatal', 'SWC']
    autonomous = ['WUG', 'WUA', 'Water Board', 'Trust']
    private = ['Company', 'Private operator']
    other = ['None', 'Other']

    final_df_test['authority_scheme'] = final_df_test['scheme_management']
    final_df_test.loc[final_df_test['authority_scheme'].isin(non), 'authority_scheme'] = 'non-autonomous'
    final_df_test.loc[final_df_test['authority_scheme'].isin(autonomous), 'authority_scheme'] = 'autonomous'
    final_df_test.loc[final_df_test['authority_scheme'].isin(private), 'authority_scheme'] = 'private'
    final_df_test.loc[final_df_test['authority_scheme'].isin(other), 'authority_scheme'] = 'other'

    # keep source, but the rare classes will be put together
    other = ['other', 'unknown']
    final_df_test['source'] = np.where(final_df_test['source'] == 'unknown', 'other', final_df_test['source'])

    # Drop during EDA I already decided what features to keep and which ones to drop
    final_df_test = final_df_test.drop(
        ['Unnamed: 0', 'id', 'amount_tsh', 'date_recorded', 'wpt_name', 'num_private', 'subvillage', 'region',
         'district_code', 'lga', 'ward', 'recorded_by', 'scheme_name', 'extraction_type_group', 'management',
         'management_group', 'payment', 'quality_group', 'quantity_group', 'source_class', 'source_type',
         'waterpoint_type_group', 'construction_year'], axis=1)

    return final_df_test


cat_col = ['funder', 'installer', 'basin', 'extraction_type', 'extraction_type_class', 'payment_type', 'water_quality',
           'quantity', 'source', 'waterpoint_type', 'scheme_management', 'season', 'region_district',
           'authority_scheme']
num_col = ['gps_height', 'longitude', 'latitude', 'population', 'public_meeting', 'age', 'permit', 'amount_tsh_missing',
           'region_code', ]



X = train_values.copy()
XT = test_values.copy()
temp = train_labels.copy()
target_status_group = {'functional': 0,
                           'non functional': 2,
                           'functional needs repair': 1}
temp['status_group'] = temp['status_group'].replace(target_status_group)

Y = temp['status_group'].copy()

data = final_fun_2(X)
test_data = final_fun_2(XT)


from pandas.core.common import random_state
lgbm1 = LGBMClassifier(colsample_bytree=0.8, max_depth=25,
                                min_split_gain=0.3, n_estimators=400,
                                num_leaves=100,  reg_alpha=1.3,
                                reg_lambda=1.1, subsample=0.8,
                                subsample_freq=20,random_state = 42)

lgbm2 = LGBMClassifier(colsample_bytree=0.8, max_depth=25,
                                min_split_gain=0.3, n_estimators=400,
                                num_leaves=100,  reg_alpha=1.3,
                                reg_lambda=1.1, subsample=0.8,
                                subsample_freq=20,random_state = 68)

lgbm3 = LGBMClassifier(colsample_bytree=0.8, max_depth=25,
                                min_split_gain=0.3, n_estimators=400,
                                num_leaves=100, reg_alpha=1.3,
                                reg_lambda=1.1, subsample=0.8,
                                subsample_freq=20,random_state = 70)

lgbm4 = LGBMClassifier(colsample_bytree=0.8, max_depth=25,
                                min_split_gain=0.3, n_estimators=400,
                                num_leaves=100, reg_alpha=1.3,
                                reg_lambda=1.1, subsample=0.8,
                                subsample_freq=20,random_state = 912)

lgbm5 = LGBMClassifier(colsample_bytree=0.8, max_depth=25,
                                min_split_gain=0.3, n_estimators=400,
                                num_leaves=100,  reg_alpha=1.3,
                                reg_lambda=1.1, subsample=0.8,
                                subsample_freq=20,random_state = 5)



vc = VotingClassifier([('lgbm1', lgbm1), ('lgbm2', lgbm2), ('lgbm3', lgbm3), ('lgbm4', lgbm4), ('lgbm5', lgbm5)],
                      voting='soft')

scaler = MinMaxScaler()
encoder = My_LabelEncoder()

# putting numeric columns to scaler and categorical to encoder
num_transformer = make_pipeline(scaler)
cat_transformer = make_pipeline(encoder)

# getting together our scaler and encoder with preprocessor
preprocessor = ColumnTransformer(
    transformers=[('num', num_transformer, num_col),
                  ('cat', cat_transformer, cat_col)])

# giving all values to pipeline

pipe = make_pipeline(preprocessor, vc)

# fit and predict
pipe.fit(data, Y)

y_pred = pipe.predict(test_data)

y_pred_train = pipe.predict(data)

# print best model scores on test data
# print best model scores on test data
print("Accuracy score train: {}".format(accuracy_score(Y, y_pred_train)))
print('-' * 20)
print("Balance Accuracy score train: {}".format(balanced_accuracy_score(Y, y_pred_train)))
print('-' * 20)
print("micro avg score train: {}".format(f1_score(Y, y_pred_train, average='micro')))
print(classification_report(Y, y_pred_train))

sub = pd.DataFrame(y_pred_train, columns=['status_group'])

target_status_group = {0: 'functional',
                           2: 'non functional',
                           1: 'functional needs repair'}
sub['status_group'] = sub['status_group'].replace(target_status_group)

dump(pipe, 'clf.joblib')
print('done')
