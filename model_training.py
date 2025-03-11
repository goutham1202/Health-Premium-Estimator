# # import pandas as pd
# # from sklearn.linear_model import LinearRegression
# # from sklearn.svm import SVR
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn import metrics
# # data = pd.read_csv("E:\\1111 CMR UNIVERSITY\\ML Project\\insurance.csv")
# # from sklearn.model_selection import train_test_split
# # print("Number of Rows",data.shape[0])
# # print("Number of Columns",data.shape[1])
# # X = data.drop(['PremiumPrice'],axis=1)
# # y = data['PremiumPrice']

# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# # lr = LinearRegression()
# # lr.fit(X_train,y_train)
# # svm = SVR()
# # svm.fit(X_train,y_train)
# # rf = RandomForestRegressor()
# # rf.fit(X_train,y_train)
# # gr = GradientBoostingRegressor()
# # gr.fit(X_train,y_train)
# # y_pred1 = lr.predict(X_test)
# # y_pred2 = svm.predict(X_test)
# # y_pred3 = rf.predict(X_test)
# # y_pred4 = gr.predict(X_test)

# # df1 = pd.DataFrame({'Actual':y_test,'Lr':y_pred1,
# #                   'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})
# # score1 = metrics.r2_score(y_test,y_pred1)
# # score2 = metrics.r2_score(y_test,y_pred2)
# # score3 = metrics.r2_score(y_test,y_pred3)
# # score4 = metrics.r2_score(y_test,y_pred4)
# # s1 = metrics.mean_absolute_error(y_test,y_pred1)
# # s2 = metrics.mean_absolute_error(y_test,y_pred2)
# # s3 = metrics.mean_absolute_error(y_test,y_pred3)
# # s4 = metrics.mean_absolute_error(y_test,y_pred4)
# # # Assuming you have code and calculations above this point in the notebook

# # # Get user input for the features
# # age = float(input("Enter your age: "))
# # height = float(input("Enter your height in centimeters: "))
# # weight = float(input("Enter your weight in kilograms: "))
# # diabetes = int(input("Do you have diabetes? (0 for No, 1 for Yes): "))
# # blood_pressure_problems = int(input("Do you have blood pressure problems? (0 for No, 1 for Yes): "))

# # # Create a dictionary with the user-input data
# # user_data = {
# #     'Age': age,
# #     'Diabetes': diabetes,
# #     'BloodPressureProblems': blood_pressure_problems,
# #     'Height': height,
# #     'Weight': weight
# # }

# # user_data

# # df = pd.DataFrame(user_data,index=[0])
# # new_pred = gr.predict(df)
# # print("You Premium estimation is : ",new_pred[0])
# # gr = GradientBoostingRegressor()
# # gr.fit(X,y)
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load dataset
# data = pd.read_csv("E:\\1111 CMR UNIVERSITY\\ML Project\\insurance.csv")

# # Display dataset details
# print("Number of Rows:", data.shape[0])
# print("Number of Columns:", data.shape[1])

# # Splitting data
# X = data.drop(['PremiumPrice'], axis=1)
# y = data['PremiumPrice']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature scaling for SVR
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Model Training
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# svm = SVR()
# svm.fit(X_train_scaled, y_train)  # SVR requires scaled data

# rf = RandomForestRegressor(random_state=42)
# rf.fit(X_train, y_train)

# gr = GradientBoostingRegressor(random_state=42)
# gr.fit(X_train, y_train)

# # Model Predictions
# y_pred1 = lr.predict(X_test)
# y_pred2 = svm.predict(X_test_scaled)  # Use scaled data for SVR
# y_pred3 = rf.predict(X_test)
# y_pred4 = gr.predict(X_test)

# # Evaluate Models
# print("\nModel Performance Metrics:")
# print(f"Linear Regression R² Score: {metrics.r2_score(y_test, y_pred1):.4f}")
# print(f"SVR R² Score: {metrics.r2_score(y_test, y_pred2):.4f}")
# print(f"Random Forest R² Score: {metrics.r2_score(y_test, y_pred3):.4f}")
# print(f"Gradient Boosting R² Score: {metrics.r2_score(y_test, y_pred4):.4f}")

# print("\nMean Absolute Error (MAE):")
# print(f"Linear Regression: {metrics.mean_absolute_error(y_test, y_pred1):.2f}")
# print(f"SVR: {metrics.mean_absolute_error(y_test, y_pred2):.2f}")
# print(f"Random Forest: {metrics.mean_absolute_error(y_test, y_pred3):.2f}")
# print(f"Gradient Boosting: {metrics.mean_absolute_error(y_test, y_pred4):.2f}")

# # Get user input for the features
# print("\nEnter your details for premium estimation:")
# age = float(input("Enter your age: "))
# height = float(input("Enter your height in centimeters: "))
# weight = float(input("Enter your weight in kilograms: "))
# diabetes = int(input("Do you have diabetes? (0 for No, 1 for Yes): "))
# blood_pressure_problems = int(input("Do you have blood pressure problems? (0 for No, 1 for Yes): "))

# # Prepare user input data
# user_data = pd.DataFrame({
#     'Age': [age],
#     'Diabetes': [diabetes],
#     'BloodPressureProblems': [blood_pressure_problems],
#     'Height': [height],
#     'Weight': [weight]
# })

# # Predict premium using the best model (Gradient Boosting)
# new_pred = gr.predict(user_data)
# formatted_premium = f"{round(new_pred[0]):,}"
# print("\nYour estimated premium price is: ₹", formatted_premium)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib  # To save/load model

# Load dataset
data = pd.read_csv("insurance.csv")

# Prepare data
X = data.drop(['PremiumPrice'], axis=1)
y = data['PremiumPrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gr = GradientBoostingRegressor()
gr.fit(X_train, y_train)

# Save the trained model
joblib.dump(gr, "premium_model.pkl")
print("Model saved successfully!")
