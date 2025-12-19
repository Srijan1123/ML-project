import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

user_df = pd.read_csv(r"C:\Users\Acer\Downloads\kaggle\sales\users_interactions.csv")
articles_df = pd.read_csv(r"C:\Users\Acer\Downloads\kaggle\sales\shared_articles.csv")

'''print(user_df.head())
print(articles_df.head())

print(user_df.info())
print(articles_df.info())

print(user_df.isnull().sum())
print(articles_df.isnull().sum())'''


#-------Let's work on the core things here------
#1 - who is reading the content
#2 - what are they doing
#3 -  how often and from where
#4 -- At last let's make the recommendation logic

#-----WHo is reading------

# Total unique users
total_users = user_df['personId'].nunique()
print("Total unique users:", total_users)

# Interactions per user
user_interactions = user_df['personId'].value_counts()

print("\nTop 10 most active users:")
print(user_interactions.head(10))

# statistics
print("\nUser interaction statistics:")
print(user_interactions.describe())

user_interactions.head(10).plot(kind='bar',figsize=(10,5))
plt.title("Top 10 most active user")
plt.xlabel("userId")
plt.ylabel("Number of interaction")
plt.show()



#------what are they doing--------
article_views = (
    user_df[user_df['eventType'] == 'VIEW']
    ['contentId']
    .value_counts()
)

print("Top 10 most read articles (by contentId):")
print(article_views.head(10))

article_views.head(10).plot(kind='bar', figsize=(10,5))
plt.title("Top 10 Most Read Articles")
plt.xlabel("Content ID")
plt.ylabel("Number of Views")
plt.show()



#---------how often and from where----
user_df ['timestamp'] = pd.to_datetime(user_df['timestamp'])


#-------Per day-------
daily_activity = user_df.groupby(user_df['timestamp'].dt.date).size()
print("daily activity")
print(daily_activity.head(10))

#---------Per hour------
hourly_activity = user_df.groupby(user_df['timestamp'].dt.hour).size()
print("Hourly activity")
print(hourly_activity.head(14))


# ------interaction per country----
country_activity = user_df['userCountry'].value_counts()
print("Interactions per country")
print(country_activity.head(10))


daily_activity.plot(kind='line', figsize=(12,5))
plt.title("Daily User Activity")
plt.xlabel("Date")
plt.ylabel("Number of Interactions")
plt.show()

hourly_activity.plot(kind='bar', figsize=(12,5))
plt.title("Hourly User Activity")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Interactions")
plt.show()

country_activity.head(10).plot(kind='bar', figsize=(12,5))
plt.title("Top 10 Countries by User Activity")
plt.xlabel("Country")
plt.ylabel("Number of Interactions")
plt.show()




#--------Recommendation logic-------
top_articles = article_views.head(20).index.tolist()
def recommend_articles(user_id, n=5):
    read_articles = user_df[user_df['personId'] == user_id]['contentId'].tolist()
    recommendations = [a for a in top_articles if a not in read_articles]
    return recommendations[:n]












