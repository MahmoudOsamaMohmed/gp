#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd


# In[2]:


Properties_links = []
for page in range(200):
    response = requests.get(f'https://www.dubizzle.com.eg/en/properties/apartments-duplex-for-sale/cairo/?page={page}')
    soup = BeautifulSoup(response.text, "html.parser")
    print(page)
    for Property in soup.find_all('li',class_='undefined'):
        try:
            Property_link = 'https://www.dubizzle.com.eg/'+Property.find('div',class_='_38ab2099').select('a')[0].get('href')
        except:
            Property_link = ''
        
        Properties_links.append(Property_link)
print('The Scraping Is Done')


# In[3]:


for page in range(200):
    response = requests.get(f'https://www.dubizzle.com.eg/en/properties/apartments-duplex-for-sale/giza/?page={page}')
    soup = BeautifulSoup(response.text, "html.parser")
    print(page)
    for Property in soup.find_all('li',class_='undefined'):
        try:
            Property_link = 'https://www.dubizzle.com.eg/'+Property.find('div',class_='_38ab2099').select('a')[0].get('href')
        except:
            Property_link = ''
        
        Properties_links.append(Property_link)
print('The Scraping Is Done')


# In[4]:


for page in range(200):
    response = requests.get(f'https://www.dubizzle.com.eg/en/properties/apartments-duplex-for-sale/alexandria/?page={page}')
    soup = BeautifulSoup(response.text, "html.parser")
    print(page)
    for Property in soup.find_all('li',class_='undefined'):
        try:
            Property_link = 'https://www.dubizzle.com.eg/'+Property.find('div',class_='_38ab2099').select('a')[0].get('href')
        except:
            Property_link = ''
        
        Properties_links.append(Property_link)
print('The Scraping Is Done')


# In[5]:


len(Properties_links)
mylinks = set(Properties_links)
len(mylinks)


# In[6]:


houes = []
Properties=[]
i=0
for link in mylinks:
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")
    i=i+1
    print(i)
    Type = 'Unknown'
    Price = 'Unknown'
    Bedrooms = 'Unknown'
    Bathrooms = 'Unknown'
    Area = 'Unknown'
    Furnished = 'Unknown'
    Level = 'Unknown'
    Payment_Option = 'Unknown'
    Delivery_Term = 'Unknown '
    try:
        City = soup.find('span',class_='_34a7409b').text.split(',')[0]
    except:
        City = ''
    for elemnt in soup.find_all('div',class_='b44ca0b3'):
        if elemnt.select('span')[0].text == 'Type' :
            Type = elemnt.select('span')[1].text
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Price' :
            try:
                Price = int(elemnt.select('span')[1].text.replace(',',''))
            except:
                Price = ''
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Bedrooms' :
            try:
                Bedrooms = int(elemnt.select('span')[1].text)
            except:
                Bedrooms = ''
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Bathrooms' :
            try:
                Bathrooms = int(elemnt.select('span')[1].text)
            except:
                Bathrooms = ''
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Area (m²)' :
            try:
                Area = float(elemnt.select('span')[1].text)
            except:
                Area = ''
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Furnished' :
            Furnished = elemnt.select('span')[1].text
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Level' :
            Level = elemnt.select('span')[1].text
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Payment Option' :
            Payment_Option = elemnt.select('span')[1].text
        else :
            pass
            #########################################
        if elemnt.select('span')[0].text == 'Delivery Term' :
            Delivery_Term = elemnt.select('span')[1].text
        else :
            pass
    houes=[Type,Price,Bedrooms,Bathrooms,Area,Furnished,Level,Payment_Option,Delivery_Term,City]
    Properties.append(houes)
print('The Scraping Is Done')


# In[7]:


Details = ['Type','Price','Bedrooms','Bathrooms','Area','Furnished','Level','Payment_Option','Delivery_Term','City']  
with open('Properties_links_olx.csv', 'w', encoding='utf-8', newline ='') as file:
    write = csv.writer(file) 
    write.writerow(Details) 
    write.writerows(Properties)

