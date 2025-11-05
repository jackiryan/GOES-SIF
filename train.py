"""
Training code for model - by Otto Lamminpää
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.io import loadmat
# import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

#datafile = "oco3_0p01d_202006_lsr_par_clean.csv"
datafile1 = "new_oco3_0p01d_202006_lsr_par_clean.csv"
datafile2 = "new_oco3_0p01d_202106_lsr_par_clean.csv"
df1 = pd.read_csv(datafile1)
df2 = pd.read_csv(datafile2)
df = pd.concat([df1,df2])

features = df.iloc[:,:]
features.drop(columns='sif_740nm')
labels = df['sif_740nm']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=features['IGBP_Primary'], random_state=40)

# take out a box
lonmin = -95 
lonmax = -85
latmin = 32
latmax = 37

box = X_train[(lonmin  < X_train['longitude']) & (X_train['longitude'] < lonmax)]
box = box[(latmin < box['latitude']) & (box['latitude'] < latmax)]
box = box[box.date < 20210000]

X_train = X_train[~X_train.index.isin(box.index)]
X_test = X_test[~X_test.index.isin(box.index)]

#y_test = pd.concat([y_test, y_train[y_train.index.isin(box.index)]])
y_train = y_train[~y_train.index.isin(box.index)]
y_test = y_test[~y_test.index.isin(box.index)]

X_tr = X_train.iloc[:,5:]
X_te = X_test.iloc[:,5:]

# Define the hyperparameter grid
param_distributions = {
    'learning_rate': np.linspace(0.01, 0.3, 30),   # Learning rates from 0.01 to 0.3
    'max_depth': list(range(3, 13)),              # Max depths from 3 to 12
    'n_estimators': list(range(50, 201)),         # Number of trees from 50 to 200
    'subsample': np.linspace(0.5, 1, 51)          # Subsample ratios from 0.5 to 1
}

# Initialize the base model
base_model = xgb.XGBRegressor()

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_distributions,
    n_iter=100,    # Number of parameter settings that are sampled
    cv=5,          # 5-fold cross-validation
    verbose=1,
    random_state=40,
    n_jobs=-1      # Use all available cores
)

# Fit the model
random_search.fit(X_tr, y_train)

# Best hyperparameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

y_predicted = best_model.predict(X_te)
y_predicted_box = best_model.predict(box.iloc[:,5:])

print(f"Best hyperparameters: {best_params}")

# SHAP explanation: this part didn't work properly so commenting out for now
'''
explainer = shap.Explainer(best_model)
shap_values = explainer(features)

# Function to visualize SHAP values
def plot_shap_summary(shap_values, features):
    plt.figure()
    shap.summary_plot(shap_values, features)
    plt.title("SHAP Summary Plot for XGBoost Model")
    plt.show()

# Plot SHAP summary
plot_shap_summary(shap_values, features)
'''

xline1 = [lonmin, lonmax]
yline1 = [latmin, latmin]
xline2 = [lonmin, lonmax]
yline2 = [latmax, latmax]
xline3 = [lonmin, lonmin]
yline3 = [latmin, latmax]
xline4 = [lonmax, lonmax]
yline4 = [latmin, latmax]


rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r_squared = r2_score(y_test, y_predicted)
x_line = np.linspace(-2, 5, 100)
plt.hist2d(y_test, y_predicted, bins=300, cmap='nipy_spectral')
plt.colorbar(label='Density') 
plt.plot(x_line, x_line,color='white', linestyle='--', linewidth=2, label='1-to-1')
plt.xlabel("OCO-3 SIF [W/$\mathrm{m}^2$/sr/μm]")
plt.ylabel("Predicted SIF [W/$\mathrm{m}^2$/sr/μm]")
plt.xlim(-1,3.5)
plt.ylim(-1,3.5)
plt.annotate("RMSE " + str(round(rmse,4)) + "\n$R^2$: "+str(round(r_squared,4)), xy=(0,0),xytext=(-0.5, 2.5), color='white',
            fontsize=12)
plt.show()


rmse = np.sqrt(mean_squared_error(box['sif_740nm'], y_predicted_box))
r_squared = r2_score(box['sif_740nm'], y_predicted_box)
x_line = np.linspace(0, 5, 100)
plt.hist2d(box['sif_740nm'], y_predicted_box, bins=200, cmap='nipy_spectral')
plt.colorbar(label='Density') 
plt.plot(x_line, x_line,color='white', linestyle='--', linewidth=2, label='1-to-1')
plt.xlabel("OCO-3 SIF [W/$\mathrm{m}^2$/sr/μm]")
plt.ylabel("Predicted SIF [W/$\mathrm{m}^2$/sr/μm]")
plt.xlim(0,3)
plt.ylim(0,3)
plt.annotate("RMSE " + str(round(rmse,4)) + "\n$R^2$: "+str(round(r_squared,4)), xy=(0,0),xytext=(0.5, 2.5), color='white',
            fontsize=12)
plt.show()


plt.hist2d(X_train["longitude"],X_train["latitude"], bins=300, cmap='nipy_spectral')
plt.colorbar(label='Data Count') 
plt.plot(xline1,yline1,color='white')
plt.plot(xline2,yline2,color='white')
plt.plot(xline3,yline3,color='white')
plt.plot(xline4,yline4,color='white')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Training Data Distribution")
plt.show()

plt.hist2d(X_test["longitude"],X_test["latitude"], bins=300, cmap='nipy_spectral')
plt.colorbar(label='Data Count') 
plt.plot(xline1,yline1,color='white')
plt.plot(xline2,yline2,color='white')
plt.plot(xline3,yline3,color='white')
plt.plot(xline4,yline4,color='white')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Test Data Distribution")
plt.show()

lo = X_test["longitude"]
la = X_test["latitude"]
si = y_test.values
plt.scatter(lo,la,s=1,c=si,cmap='Greens',vmin=-1,vmax=6)
plt.colorbar(label='SIF [W/$\mathrm{m}^2$/sr/μm]') 
plt.plot(xline1,yline1,color='black')
plt.plot(xline2,yline2,color='black')
plt.plot(xline3,yline3,color='black')
plt.plot(xline4,yline4,color='black')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Test SIF Values")
plt.show()

lo = X_test["longitude"]
la = X_test["latitude"]
si = y_predicted 
plt.scatter(lo,la,s=1,c=si,cmap='Greens',vmin=-1,vmax=6)
plt.colorbar(label='SIF [W/$\mathrm{m}^2$/sr/μm]') 
plt.plot(xline1,yline1,color='black')
plt.plot(xline2,yline2,color='black')
plt.plot(xline3,yline3,color='black')
plt.plot(xline4,yline4,color='black')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted SIF Values")
plt.show()

lo = X_test["longitude"]
la = X_test["latitude"]
si = np.abs(y_test.values - y_predicted)
plt.scatter(lo,la,s=1,c=si,cmap='nipy_spectral') #'Spectral',vmin=-2,vmax=2)
plt.colorbar(label='SIF error [W/$\mathrm{m}^2$/sr/μm]') 
plt.plot(xline1,yline1,color='black')
plt.plot(xline2,yline2,color='black')
plt.plot(xline3,yline3,color='black')
plt.plot(xline4,yline4,color='black')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted SIF Error")
plt.show()

plt.scatter(box['longitude'],box['latitude'],c=box['sif_740nm'],s=1,cmap='Greens',vmin=-1,vmax=6)
plt.colorbar(label='SIF [W/$\mathrm{m}^2$/sr/μm]') 
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Test SIF Values")
plt.show()

plt.scatter(box['longitude'],box['latitude'],c=y_predicted_box,s=1,cmap='Greens',vmin=-1,vmax=6)
plt.colorbar(label='SIF [W/$\mathrm{m}^2$/sr/μm]') 
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted SIF Values")
plt.show()


plt.scatter(box['longitude'],box['latitude'],c=abs(box['sif_740nm'] - y_predicted_box),s=1,cmap='nipy_spectral')
plt.colorbar(label='SIF error [W/$\mathrm{m}^2$/sr/μm]') 
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted SIF Absolute Error")
plt.show()


data_all = X_test.copy()
data_all['SIF_true'] = y_test.values
data_all['SIF_predicted'] = y_predicted
data_all['SIF_error'] = y_test.values - y_predicted

data_all.boxplot(column='SIF_error', by='IGBP_Primary')
plt.show()