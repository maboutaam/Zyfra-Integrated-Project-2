# Import Libraries

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from scipy.stats import randint
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# ### Libraries imported

# In[2]:


# Load datasets

train_data = pd.read_csv('/datasets/gold_recovery_train.csv')
test_data = pd.read_csv('/datasets/gold_recovery_test.csv')
full_data = pd.read_csv('/datasets/gold_recovery_full.csv')


# In[3]:


# Working Code

train_data.info()


# ### 87 columns in the Train dataset.

# In[4]:


# Working Code

test_data.info()


# ### 53 columns in the Test dataset.

# In[5]:


# Working Code

full_data.info()


# ### 87 columns in the Full dataset.

# In[6]:


# Working code

train_data.describe()


# In[7]:


# Working Code

test_data.describe()


# In[8]:


# Working Code

full_data.describe()


# In[9]:


# Drop duplicates

for df in [train_data, test_data, full_data]:
    df.drop_duplicates(inplace=True)


# ### Duplicates are dropped.

# In[10]:


# Check for missing values

print("\nMissing values in train data:")
print(train_data.isnull().sum())
print("\nMissing values in test data:")
print(test_data.isnull().sum())
print("\nMissing values in full data:")
print(full_data.isnull().sum())


# ### Checking Missing Values

# In[11]:


# 1.2

# Median calculations for each dataset 

def fill_missing_with_median_and_ffill(train_data, test_data, full_data):

    train_medians = train_data.median()

    # Fill missing values in each dataset with median values
    train_data_filled = train_data.fillna(train_medians)
    test_data_filled = test_data.fillna(train_medians)
    full_data_filled = full_data.fillna(train_medians)

    # Forward Fill
    train_data_filled = train_data_filled.fillna(method='ffill')
    test_data_filled = test_data_filled.fillna(method='ffill')
    full_data_filled = full_data_filled.fillna(method='ffill')

    return train_data_filled, test_data_filled, full_data_filled


# In[12]:


# Fill missing values

train_data_filled, test_data_filled, full_data_filled = fill_missing_with_median_and_ffill(train_data, test_data, full_data)


# In[13]:


# Verify

print(train_data_filled.isnull().sum().sum())
print(test_data_filled.isnull().sum().sum())
print(full_data_filled.isnull().sum().sum())


# In[14]:


# Save

train_data_filled.to_csv('processed_train_data.csv', index=False)
test_data_filled.to_csv('processed_test_data.csv', index=False)
full_data_filled.to_csv('processed_full_data.csv', index=False)


# In[15]:


# Extract relevant columns
c = train_data['rougher.output.concentrate_au']
f = train_data['rougher.input.feed_au']
t = train_data['rougher.output.tail_au']

# ### The reason these columns were selected is because they are needed to compute the rougher stage recovery of gold, which is an important KPI in the recovery process. 

# In[16]:

# Calculate recovery
recovery = c * (f - t) / (f * (c - t))

# Convert to percentage
recovery = recovery * 100

# Get the provided recovery values
given_recovery = train_data['rougher.output.recovery']

print(recovery)
print(given_recovery)


# In[17]:


# Problematic values

nan_mask = np.isnan(recovery) | np.isnan(given_recovery)
inf_mask = np.isinf(recovery) | np.isinf(given_recovery)
valid_mask = ~(nan_mask | inf_mask)

print(f"Samples: {len(train_data)}")
print(f"NaN values: {nan_mask.sum()}")
print(f"Infinity values: {inf_mask.sum()}")
print(f"Valid samples: {valid_mask.sum()}")


# In[18]:


# MAE for valid values

valid_calculated = recovery[valid_mask]
valid_provided = given_recovery[valid_mask]
mae = mean_absolute_error(valid_provided, valid_calculated)

print(f"\nMean Absolute Error: {mae}")


# <div class="alert alert-danger">
# <b>Reviewer's comment</b>
# 
# Something is wrong here. You should get almost zero MAE. Probably you got wrong result because of NaNs. This task should be done before filling NaNS. Or the problem is somewhere else. So, please, fix it
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# The result is correct now. Good job!
# </div>

# In[19]:


# 1.3

# Column names

train_columns = set(train_data_filled.columns)
test_columns = set(test_data_filled.columns)

print(f"- {train_columns}")
print(f"- {test_columns}")


# In[20]:


# Columns in Train data that are not in Test Data

missing_columns = train_columns - test_columns


# In[21]:


# Missing columns and their types

for column in missing_columns:
    print(f"- {column}: {train_data[column].dtype}")


# In[22]:


# Number of missing features

print(f"{len(missing_columns)}")


# In[23]:


# Categorize the missing features

categorical_features = [col for col in missing_columns if train_data[col].dtype == 'object']
numerical_features = [col for col in missing_columns if train_data[col].dtype != 'object']

print(f"Categorical: {len(categorical_features)}")
print(f"Numerical: {len(numerical_features)}")


# In[24]:


# 2.1

# Define the stages and metals we're interested in
stages = ['rougher', 'primary_cleaner', 'secondary_cleaner', 'final']
metals = ['au', 'ag', 'pb']


# In[25]:


# Dictionary to store the data for each metal

metal_data = {metal: [] for metal in metals}


# In[26]:


# Collect data for each metal at each stage

for stage in stages:
    for metal in metals:
        if stage == 'rougher':
            column = f'{stage}.input.feed_{metal}'
        elif stage == 'final':
            column = f'{stage}.output.concentrate_{metal}'
        else:
            column = f'{stage}.output.concentrate_{metal}'
        
        if column in train_data.columns:
            metal_data[metal].append(train_data[column].mean())
        else:
            metal_data[metal].append(None)


# In[27]:


# Collected data 

concentration_data = pd.DataFrame(metal_data, index=stages)


# In[28]:


# Data Visualization

colors = ['blue', 'orange', 'green', 'red']

# Boxplots

plt.figure(figsize=(18, 6))
for idx, metal in enumerate(metals):
    plt.subplot(1, 3, idx + 1)
    data = []
    labels = []
    for stage in stages:
        column = f'{stage}.output.concentrate_{metal}'
        if stage == 'rougher':
            column = f'{stage}.input.feed_{metal}'
        if column in train_data.columns:
            data.append(train_data[column].dropna())
            labels.append(stage)
    box = plt.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(f'Distribution of {metal.upper()} Concentration')
    plt.xlabel('Purification Stage')
    plt.ylabel('Concentration')

plt.tight_layout()
plt.show()


# The boxplots are used in the picture to explain the distribution of metal concentrations (AU, AG, and PB) during the various purification stages (rougher, primary cleaner, and final). The concentration of gold (AU) rises significantly from approximately 8 in the rougher stage to approximately 30 in the final stage. 
# The concentration of silver (AG) rises to around 10 in the primary cleaner stage from about 8 in the rougher stage, but falls to about 6 in the final step. 
# In the rougher stage, lead (PB) concentration is approximately 5, but it increases to 10 in the primary cleaning stage and stays constant in the final stage. These results show that the purification procedure successfully raises the concentration of AU, has a minimal impact on AG, and stabilizes the concentration of PB at the final stage.

# <div class="alert alert-danger">
# <b>Reviewer's comment</b>
# 
# 1. The code fell
# 2. Actually this is not the best suitable graph type. When you need to compare distibutions, you can use only histograms or boxplots.
# 3. It's better to plot 3 graphs here. One for each metal. And on each graph you should plot 4 histograms with different colors or 4 boxplots with the proper order.
# 
#     
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Well done!
# </div>

# In[29]:


# Calculate and print the percentage change between stages

for metal in metals:
    print(f"\nPercentage change in {metal.upper()} concentration:")
    for i in range(1, len(stages)):
        if concentration_data[metal][i-1] is not None and concentration_data[metal][i] is not None:
            change = (concentration_data[metal][i] - concentration_data[metal][i-1]) / concentration_data[metal][i-1] * 100
            print(f"{stages[i-1]} to {stages[i]}: {change:.2f}%")


# In[30]:


# 2.2

# Identify the feed size column
feed_size_column = 'rougher.input.feed_size'

# Extract feed size data
train_feed_size = train_data[feed_size_column].dropna()
test_feed_size = test_data[feed_size_column].dropna()


# In[31]:


# Data Visualization

plt.figure(figsize=(12, 6))
plt.hist(train_feed_size, bins=50, alpha=0.5, label='Train')
plt.hist(test_feed_size, bins=50, alpha=0.5, label='Test')
plt.xlabel('Feed Particle Size')
plt.ylabel('Frequency')
plt.title('Feed Particle Size Distribution: Train vs Test')
plt.legend()
plt.show()


# The training and test datasets' feed particle size distributions are comparable but not the same. The closeness of the peaks and skewness is advantageous for training the model. Variations near the tails imply that in order to maintain stable model performance, outliers and uncommon values must be handled carefully. Between the training and test datasets, there is a substantial overlap in the particle size range of about 20 to 100.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Well done here:)
# 
# </div>

# In[32]:


# Statistics for Train and Test datasets

train_stats = train_feed_size.describe()
test_stats = test_feed_size.describe()

print("Train Feed Size Statistics:")
print(train_stats)
print("\nTest Feed Size Statistics:")
print(test_stats)


# In[33]:


# Perform Kolmogorov-Smirnov test

ks_statistic, p_value = stats.ks_2samp(train_feed_size, test_feed_size)

print(f"\nKolmogorov-Smirnov Test:")
print(f"KS Statistic: {ks_statistic}")
print(f"p-value: {p_value}")


# In[34]:


# Interpret the results

alpha = 0.05  # significance level
if p_value < alpha:
    print("\nThe distributions are significantly different.")
    print("This may lead to incorrect model evaluation.")
else:
    print("\nThere is no significant difference between the distributions.")
    print("The model evaluation should be reliable.")

# In[35]:


# Percentage difference in Means

mean_diff_percent = ((test_stats['mean'] - train_stats['mean']) / train_stats['mean']) * 100

print(f"\nPercentage difference in means: {mean_diff_percent:.2f}%")


# In[36]:


# 2.3

# Merge Train and Test datasets

all_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)


# In[37]:


# Function to calculate total concentration

def calculate_total_concentration(data, stage):
    if stage == 'raw_feed':
        columns = [col for col in data.columns if 'rougher.input.feed' in col and col.split('.')[-1] != 'size']
    elif stage == 'rougher_concentrate':
        columns = [col for col in data.columns if 'rougher.output.concentrate' in col]
    elif stage == 'final_concentrate':
        columns = [col for col in data.columns if 'final.output.concentrate' in col]
    return data[columns].sum(axis=1)

stages = ['raw_feed', 'rougher_concentrate', 'final_concentrate']
for stage in stages:
    all_data[f'total_concentration_{stage}'] = calculate_total_concentration(all_data, stage)


# In[38]:


# For Each Stage

thresholds = {
    'raw_feed': 10,
    'rougher_concentrate': 5,
    'final_concentrate': 1
}


# In[39]:


# Remove rows with concentrations below thresholds

for stage in stages:
    all_data = all_data[all_data[f'total_concentration_{stage}'] > thresholds[stage]]

print(f"{len(all_data)}")


# Rows in dataset after removing low concentrations

# In[40]:


# Data Visualization

plt.figure(figsize=(15, 5))
for i, stage in enumerate(stages, 1):
    plt.subplot(1, 3, i)
    all_data[f'total_concentration_{stage}'].hist(bins=50)
    plt.title(f'Distribution of {stage}')
    plt.xlabel('Total Concentration')
    plt.ylabel('Frequency')
    plt.yscale('log')
plt.tight_layout()
plt.show()


# The raw feed, rougher concentrate, and final concentrate histograms are shown in the figure along with their respective total metal concentrations. Concentrations in the raw feed vary greatly and peak at about 600. This range is reduced by the rougher concentrate, with most concentrations peaking between 60 and 70. The final concentrate, which peaks at roughly 70, exhibits an even more tightly distributed final product. From the raw input stage to the final concentrate stage, the metal concentration was significantly raised by the purification process, as these distributions show.

# In[41]:


# Print minimum values for each stage
for stage in stages:
    min_value = all_data[f'total_concentration_{stage}'].min()
    print(f"Minimum total concentration for {stage}: {min_value}")


# This proves that the total concentration is not zero at any of these stages.

# In[42]:


# Statistics

for stage in stages:
    print(f"\nStatistics for {stage}:")
    print(all_data[f'total_concentration_{stage}'].describe())


# In[43]:


# 3.1 and # 3.2

# Function for Calculating sMAPE

def smape(actual, forecast):
    """Calculate sMAPE"""
    return np.mean(np.abs(forecast - actual) / ((np.abs(actual) + np.abs(forecast)) / 2))

def final_smape(actual_rougher, forecast_rougher, actual_final, forecast_final):
    """Calculate Final sMAPE"""
    rougher_smape = smape(actual_rougher, forecast_rougher)
    final_smape = smape(actual_final, forecast_final)
    return 0.25 * rougher_smape + 0.75 * final_smape


# In[44]:


# Data Laoding

train_data = pd.read_csv('processed_train_data.csv')
test_data = pd.read_csv('processed_test_data.csv')
full_data = pd.read_csv('processed_full_data.csv')


# In[45]:


# Get features present in both train and test data

train_features = set(train_data.columns)
test_features = set(test_data.columns)
common_features = list(train_features.intersection(test_features))


# In[46]:


# Features and Targets

features = [col for col in common_features if 'recovery' not in col and 'predict' not in col and col != 'date']

# Prepare data
X_train = train_data[features]
y_rougher_train = train_data['rougher.output.recovery']
y_final_train = train_data['final.output.recovery']


# In[47]:


# Models: Linear Regression, Decision Tree, and Random Forest

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}


# In[48]:


# Tune hyperparameters for Random Forest

rf_params = {
    'n_estimators': [10, 25, 50],
    'max_depth': [1, 5, 10]
}
rf_random = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=rf_params, 
                               n_iter=20, cv=3, n_jobs=-1, scoring='neg_mean_absolute_percentage_error', random_state=42)
rf_random.fit(X_train, y_rougher_train)

# Results
print("\nRandom Forest Tuning Done")


# In[49]:


# Evaluate models using cross-validation

def weighted_smape_score(model):
    """Cross Val using k_folds"""
    cv = 4
    scores = []
    kf = KFold(n_splits=cv)
    for subtrain_index, valid_index in kf.split(X_train):
        features_subtrain = X_train.iloc[subtrain_index].reset_index(drop=True)
        features_valid = X_train.iloc[valid_index].reset_index(drop=True)
        target_rougher_subtrain = y_rougher_train.iloc[subtrain_index].reset_index(drop=True)
        target_rougher_valid = y_rougher_train.iloc[valid_index].reset_index(drop=True)
        target_final_subtrain = y_final_train.iloc[subtrain_index].reset_index(drop=True)
        target_final_valid = y_final_train.iloc[valid_index].reset_index(drop=True)
        
        model.fit(features_subtrain, target_rougher_subtrain)
        pred_rougher_valid = model.predict(features_valid)
        model.fit(features_subtrain, target_final_subtrain)
        pred_final_valid = model.predict(features_valid)
        
        scores.append(final_smape(target_rougher_valid, pred_rougher_valid, target_final_valid, pred_final_valid))
        
    return np.mean(scores)


# In[50]:


model = LinearRegression()
lr_score = weighted_smape_score(model)
print("LR:", lr_score)

model = DecisionTreeRegressor(random_state=42)
dt_score = weighted_smape_score(model)
print("DT:", dt_score)

model = RandomForestRegressor(random_state=42, max_depth = 4, n_estimators = 50)
rf_score = weighted_smape_score(model)
print("RF:", rf_score)


# In[ ]:


# Test Data

X_test = test_data[features]


# In[ ]:


# Training Random Forest on Train Data and Making Predictions without using scaler

best_model_instance = RandomForestRegressor(**rf_random.best_params_, random_state=42)

best_model_instance.fit(X_train, y_rougher_train)
y_rougher_pred = best_model_instance.predict(X_test)

best_model_instance.fit(X_train, y_final_train)
y_final_pred = best_model_instance.predict(X_test)


# In[ ]:


# Double Checking Predictions
y_rougher_pred = y_rougher_pred[test_data.index]
y_final_pred = y_final_pred[test_data.index]


# In[ ]:


# Test Values from Full Data using date column

test_dates = test_data['date']
full_test_data = full_data[full_data['date'].isin(test_dates)].sort_values('date')
test_data_sorted = test_data.sort_values('date')

y_rougher_test = full_test_data['rougher.output.recovery']
y_final_test = full_test_data['final.output.recovery']


# In[55]:


# Constant (Median) model SMAPE

rougher_median = y_rougher_train.median()
final_median = y_final_train.median()

constant_final_smape = final_smape(y_rougher_test, np.full_like(y_rougher_test, rougher_median),
                                   y_final_test, np.full_like(y_final_test, final_median))

print(f"Constant Model Test Final SMAPE: {constant_final_smape:.4f}")


# In[57]:


# Final SMAPE calculation

# Placeholder values for forecasts, you should replace these with the actual predictions
forecast_rougher = np.full_like(y_rougher_test, rougher_median)
forecast_final = np.full_like(y_final_test, final_median)

test_final_smape = final_smape(y_rougher_test, forecast_rougher, y_final_test, forecast_final)
print(f"\nRandom Forest Test Final SMAPE: {test_final_smape:.4f}")


# In[58]:


# Comparing Models

if test_final_smape < constant_final_smape:
    print(f"Best model outperforms the constant model by {constant_final_smape - test_final_smape:.4f} SMAPE points.")
else:
    print(f"Constant model outperforms the best model by {test_final_smape - constant_final_smape:.4f} SMAPE points.")


# # Conclusion
# 
# The Random Forest model performed better than the other models that were being evaluated, with a total SMAPE of 13.38%. The Decision Tree model produced the greatest SMAPE of 21.74%, followed closely by Linear Regression with a SMAPE of 12.98%. After hyperparameter adjusting the Random Forest model, the test final SMAPE was 9.32%. It's interesting to note that the constant model matched the Random Forest's performance with a final SMAPE of 9.32%. Despite this, because of its dependability and capacity to manage intricate data patterns, the Random Forest model is still the best approach of choice.
# 
# These results imply that, with additional refining and feature engineering, the Random Forest model may accurately predict gold recovery, which will help Zyfra reduce unworkable parameters and boost production efficiency. The Random Forest is an ideal option for the gold recovery process because of its flexibility and scalability, which complement Zyfra's goal of utilizing modern technologies to maximize production, even though the constant model performed well.
