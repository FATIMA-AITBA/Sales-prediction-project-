import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv('advertising.csv')


print(df.head())         
print(df.info())         
print(df.describe())    


scaler = StandardScaler()
X = df[['TV', 'Radio', 'Newspaper']]  
y = df['Sales']                     

X_scaled = scaler.fit_transform(X)   


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Random Forest - Mean Squared Error: {mse}')
print(f'Random Forest - R^2 Score: {r2}')

joblib.dump(rf_model, 'model/random_forest_model.pkl')

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red') 
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales with random forest model')
plt.show()