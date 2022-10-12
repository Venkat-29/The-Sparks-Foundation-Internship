#!/usr/bin/env python
# coding: utf-8

# # VENKATACHALAM P

# # Data Science and Business Analytics Intern @ The Sparks Foundation 

# ### Task 4 : Exploratory Data Analysis (EDA) - Terrorism 

# ### Dataset : globalterrorismdb_0718dist.csv [https://bit.ly/2TK5Xn5]

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('https://bit.ly/2TK5Xn5%5D')
data.head()


# In[3]:


data.columns.values


# In[4]:


data.rename(columns={'iyear':'Year','imonth':'Month','iday':"day",'gname':'Group','country_txt':'Country','region_txt':'Region','provstate':'State','city':'City','latitude':'latitude',
    'longitude':'longitude','summary':'summary','attacktype1_txt':'Attacktype','targtype1_txt':'Targettype','weaptype1_txt':'Weapon','nkill':'kill',
     'nwound':'Wound'},inplace=True)


# In[5]:


data = data[['Year','Month','day','Country','State','Region','City','latitude','longitude',"Attacktype",'kill',
               'Wound','target1','summary','Group','Targettype','Weapon','motive']]


# ### Filling the null values of the features 'Wound' and 'Kill'

# In[9]:


data['Wound'] = data['Wound'].fillna(0)
data['kill'] = data['kill'].fillna(0)


# ### Creating a new dataframe 'Casualities' with the combination of both 'Kill' and 'Wound'

# In[10]:


data['Casualities'] = data['kill'] + data['Wound']


# In[12]:


data.describe()


# ### Plotting the Number of attack in each year

# In[13]:


year = data['Year'].unique()
years_count = data['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = year,
           y = years_count,
           palette = "tab10")
plt.xticks(rotation = 50)
plt.xlabel('Attacking Year',fontsize=20)
plt.ylabel('Number of Attacks Each Year',fontsize=20)
plt.title('Attacks In Years',fontsize=30)
plt.show()


# ### Plotting the number of attacks to find the terrorist activities by region

# In[14]:


pd.crosstab(data.Year, data.Region).plot(kind='area',stacked=False,figsize=(20,10))
plt.title('Terrorist Activities By Region In Each Year',fontsize=25)
plt.ylabel('Number of Attacks',fontsize=20)
plt.xlabel("Year",fontsize=20)
plt.show()


# In[15]:


attack = data.Country.value_counts()[:10]
attack


# In[16]:


data.Group.value_counts()[1:10]


# ### Plotting the top countries affected

# In[17]:


plt.subplots(figsize=(20,10))
sns.barplot(data['Country'].value_counts()[:10].index,data['Country'].value_counts()[:10].values,palette='YlOrBr_r')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation = 50)
plt.show()


# ### Plotting the people died due to attack

# In[18]:


df = data[['Year','kill']].groupby(['Year']).sum()
fig, ax4 = plt.subplots(figsize=(20,10))
df.plot(kind='bar',alpha=0.7,ax=ax4)
plt.xticks(rotation = 50)
plt.title("People Died Due To Attack",fontsize=25)
plt.ylabel("Number of killed peope",fontsize=20)
plt.xlabel('Year',fontsize=20)
top_side = ax4.spines["top"]
top_side.set_visible(False)
right_side = ax4.spines["right"]
right_side.set_visible(False)


# ### Plotting the top 10 effected city

# In[19]:


data['City'].value_counts().to_frame().sort_values('City',axis=0,ascending=False).head(10).plot(kind='bar',figsize=(20,10),color='blue')
plt.xticks(rotation = 50)
plt.xlabel("City",fontsize=15)
plt.ylabel("Number of attack",fontsize=15)
plt.title("Top 10 most effected city",fontsize=20)
plt.show()


# ### Plotting the major attacktype

# In[20]:


data['Attacktype'].value_counts().plot(kind='bar',figsize=(20,10),color='magenta')
plt.xticks(rotation = 50)
plt.xlabel("Attacktype",fontsize=15)
plt.ylabel("Number of attack",fontsize=15)
plt.title("Name of attacktype",fontsize=20)
plt.show()


# ### Plotting the number of killed

# In[21]:


data[['Attacktype','kill']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=['darkslateblue'])
plt.xticks(rotation=50)
plt.title("Number of killed ",fontsize=20)
plt.ylabel('Number of people',fontsize=15)
plt.xlabel('Attack type',fontsize=15)
plt.show()


# ### Plotting the number of wounded

# In[22]:


data[['Attacktype','Wound']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=['cyan'])
plt.xticks(rotation=50)
plt.title("Number of wounded  ",fontsize=20)
plt.ylabel('Number of people',fontsize=15)
plt.xlabel('Attack type',fontsize=15)
plt.show()


# ### Plotting the attack per year

# In[23]:


plt.subplots(figsize=(20,10))
sns.countplot(data["Targettype"],order=data['Targettype'].value_counts().index,palette="gist_heat",edgecolor=sns.color_palette("mako"));
plt.xticks(rotation=90)
plt.xlabel("Attacktype",fontsize=15)
plt.ylabel("count",fontsize=15)
plt.title("Attack per year",fontsize=20)
plt.show()


# ### Plotting the top 10 terrorist group attack

# In[25]:


data['Group'].value_counts().to_frame().drop('Unknown').head(10).plot(kind='bar',color='green',figsize=(20,10))
plt.title("Top 10 terrorist group attack",fontsize=20)
plt.xlabel("terrorist group name",fontsize=15)
plt.ylabel("Attack number",fontsize=15)
plt.show()


# ### Top 10 terroist group attack

# In[26]:


data[['Group','kill']].groupby(['Group'],axis=0).sum().drop('Unknown').sort_values('kill',ascending=False).head(10).plot(kind='bar',color='yellow',figsize=(20,10))
plt.title("Top 10 terrorist group attack",fontsize=20)
plt.xlabel("terrorist group name",fontsize=15)
plt.ylabel("No of killed people",fontsize=15)
plt.show()


# In[27]:


df=data[['Group','Country','kill']]
df=df.groupby(['Group','Country'],axis=0).sum().sort_values('kill',ascending=False).drop('Unknown').reset_index().head(10)
df


# In[28]:


kill = data.loc[:,'kill']
print('Number of people killed by terror attack:', int(sum(kill.dropna())))


# In[29]:


typeKill = data.pivot_table(columns='Attacktype', values='kill', aggfunc='sum')
typeKill


# In[30]:


countryKill = data.pivot_table(columns='Country', values='kill', aggfunc='sum')
countryKill


# **Conclusion and Results :**

# - Country with the most attacks: **Iraq**
# 
# - City with the most attacks: **Baghdad**
# 
# - Region with the most attacks: **Middle East & North Africa**
# 
# - Year with the most attacks: **2014**
# 
# - Month with the most attacks: **5**
# 
# - Group with the most attacks: **Taliban**
# 
# - Most Attack Types: **Bombing/Explosion**
# 

# ## Thank You! 
