#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from xgboost import XGBRegressor
import pandas as pd
from flask import Flask, jsonify, request

xgb = XGBRegressor()
xgb.load_model("pricing_model.json")
app = Flask(__name__)
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000/*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response
@app.route('/price', methods=['POST'])
def get_dict():
    dictionary = request.get_json()
    dictionary = pd.DataFrame(dictionary, index=[0])
    dictionary = pd.get_dummies(dictionary, columns = ['Type','Delivery_Term','Furnished','City' ,'Payment_Option'])
    Predict_price = pd.DataFrame(dictionary, index=[0], columns=xgb.get_booster().feature_names)
    Predict_price.fillna(0, inplace=True)
    price = xgb.predict(Predict_price)
    return jsonify({'value': price.tolist()})

if __name__ == '__main__':
    app.run()


# In[ ]:




