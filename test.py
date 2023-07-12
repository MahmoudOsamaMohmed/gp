#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests

BASE = "http://127.0.0.1:5000/"
Data = {"Type_Apartment":1,"Bedrooms":3,"Bathrooms":2
                 ,"Area":120,"Furnished_No":1,"Level":4
                 ,"Payment_Option_Cash":1,"Delivery_Term_Finished":1,"City_Zamalek":1}
response = requests.post(BASE + "", Data)
print(response)

