import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('C:/Users/roshn/Downloads/CR Internship/retail_price.csv')
print(data.head())

print(data.describe())


fig = px.histogram(data, 
                   x='total_price', 
                   nbins=20, 
                   title='Distribution of Total Price')
fig.show()

fig = px.box(data, 
             y='unit_price', 
             title='Box Plot of Unit Price')
fig.show()
# # Load the sales and pricing data from a CSV file
# def load_data():
#     try:
#         data = pd.read_csv('C:/Users/roshn/Downloads/CR Internship/retail_price.csv')
#         return data
#     except Exception as e:
#         messagebox.showerror("Error", str(e))

# # Perform linear regression and calculate metrics
# def optimize_price(data, target_col, feature_col):
#     try:
        
#         X = data[['qty', 'unit_price', 'comp_1', 
#           'product_score', 'comp_price_diff']]
#         y = data['total_price']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                     test_size=0.2,
#                                                     random_state=42)  

#         # Train the linear regression model
#         model = DecisionTreeRegressor()
#         model.fit(X_train, y_train)

#         # Make predictions
#         y_pred = model.predict(X)

#         # Calculate metrics
#         mse = mean_squared_error(y, y_pred)
#         r2 = r2_score(y, y_pred)

#         return model, mse, r2
#     except Exception as e:
#         messagebox.showerror("Error", str(e))

# # GUI
# def optimize_price_gui():
#     def calculate_optimal_price():
#         target_col = target_entry.get()
#         feature_col = feature_entry.get()
#         file_path = file_entry.get()

#         data = load_data('C:/Users/roshn/Downloads/CR Internship/retail_price.csv')
#         if data is not None:
#             model, mse, r2 = optimize_price(data, target_col, feature_col)

#             messagebox.showinfo("Results", f"Optimal Price: {model.intercept_} + {model.coef_[0]} * x\n\n"
#                                             f"Mean Squared Error (MSE): {mse}\n"
#                                             f"R-squared (R2): {r2}")


  

 

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
#                          marker=dict(color='blue'), 
#                          name='Predicted vs. Actual Retail Price'))
# fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
#                          mode='lines', 
#                          marker=dict(color='red'), 
#                          name='Ideal Prediction'))
# fig.update_layout(
#     title='Predicted vs. Actual Retail Price',
#     xaxis_title='Actual Retail Price',
#     yaxis_title='Predicted Retail Price'
# )
# fig.show()
