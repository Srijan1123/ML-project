import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Acer\OneDrive\Desktop\AI.ML\netflix.csv")
print(df.head())


# -------- Movies vs TV shows count -------
print("Movies and TV count")
print(df['type'].value_counts())
sns.countplot(data= df, x="type")
plt.title("Movies and tv shows")
plt.show()

# -------- Movies vs TV shows pie chart --------
count = df['type'].value_counts()

plt.pie(count, labels=count.index, autopct='%1.1f%%')
plt.title("Movies vs Tv shows")
plt.show()



# -------- Most watched genre (most frequent listed_in) -------
print("Most watched genre:")
print(df['listed_in'].value_counts().head(5))

df['listed_in'].value_counts().head(5).plot(kind="bar")
plt.title("Most watched genre")
plt.ylabel("Count")
plt.show()


# -------- Directors with most shows --------
print("Top directors")
print(df['directors'].value_counts().head(5))


# -------- Genre distribution (seaborn) --------
genre_count = df['listed_in'].value_counts().head(10)

sns.barplot(x=genre_count.values, y=genre_count.index)
plt.title("Top 10 Genre")
plt.xlabel("count")
plt.ylabel("Genre")
plt.show()


# ---------- Most common Actor ----------
print("\nMost common Actor")
df['cast'] = df['cast'].fillna("")
all_actor = df['cast'].str.split(",").explode()
actor_count = all_actor.value_counts().head(10)
print(actor_count)

actor_count.plot(kind="bar")
plt.title("Most Common Actor")
plt.xlabel("Actor")
plt.ylabel("Cont")
plt.show()



# ---------- 2. Most Frequent Genres Per Year ----------
print("\nTop Genres Per Year:")
df['listed_in'] = df['listed_in'].fillna("")
df['release_year'] = df['release_year'].astype(int)

genres_per_year = (
    df.assign(genres=df['listed_in'].str.split(", "))
      .explode("genres")
      .groupby(['release_year', 'genres'])
      .size()
      .reset_index(name="count")
)

top_genres = genres_per_year.sort_values(["release_year", "count"], ascending=[True, False]).groupby("release_year").head(1)
print(top_genres.head(10))

plt.figure(figsize=(10,5))
sns.lineplot(data=top_genres, x="release_year", y="count")
plt.title("Most Frequent Genre Per Year")
plt.show()


# ---------- 3. Countries producing most movies ----------
print("\nCountries producing movies")
df['country'] = df['country'].fillna("")
countries = df['country'].str.split(",").explode()
country_count = countries.value_counts().head(10)
print(country_count)

country_count.plot(kind="bar")
plt.title("Top 10 countries producing netflix content")
plt.show()



# ---------- 4. Trend Analysis of Movie Durations ----------
print("\nDuration Trend:")
df['duration'] = df['duration'].fillna("0 min").str.extract("(\d+)").astype(int)

plt.figure(figsize=(10,5))
sns.lineplot(data=df[df['type']=="Movie"], x="release_year", y="duration")
plt.title("Movie Duration Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Duration (minutes)")
plt.show()



# ---------- 5. Netflix Movie/Show Growth Over Time ----------
print("\nNetflix Additions per Year:")
content_trend = df.groupby(['release_year', 'type']).size().reset_index(name="count")

sns.lineplot(data=content_trend, x="release_year", y="count", hue="type")
plt.title("Netflix Content Added Over the Years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()