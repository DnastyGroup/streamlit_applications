import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.title('st.file_uploader')

st.subheader('Input CSV')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.subheader('DataFrame')
  st.write(df)
  num_of_rows = len(df)

  # Assuming 'type' is the column to categorize by
  fdf = df[df.type == "Movie"]
  num_movies = len(fdf)

  tdf = df[df.type == "TV Show"]
  num_tv_shows = len(tdf)

  st.info(f"How many titles in total?: {num_of_rows}")

  st.info(f'How many were movies?: {num_movies}')
  st.write(fdf)

  st.info(f'How many were TV shows?: {num_tv_shows}')
  st.write(tdf)

  st.info('Showing total movies vs TV shows ratio:')

  data = {"count":[num_movies,num_tv_shows], "type":["Movies", "TV Shows"]}
  df_count = pd.DataFrame(data)
  st.bar_chart(data=df_count, x="type", y="count")

  # Analysis 1: Top 5 countries with most titles
  st.subheader('Top 5 Countries with Most Titles')
  country_counts = df['country'].value_counts().head(5)
  st.bar_chart(country_counts)

  # Analysis 2: Distribution of Ratings
  st.subheader('Distribution of Ratings')
  rating_counts = df['rating'].value_counts()
  st.bar_chart(rating_counts)

  # Analysis 3: Number of Titles by Release Year
  st.subheader('Number of Titles by Release Year')
  year_counts = df['release_year'].value_counts().sort_index()
  st.line_chart(year_counts)

# Machine Learning Use Case: Titanic Survival Prediction
st.header('Machine Learning: Titanic Survival Prediction')

# Load Titanic data
titanic_df = pd.read_csv('titanic.csv')
st.subheader('Titanic Dataset')
st.write(titanic_df.head())

# EDA
st.subheader('Exploratory Data Analysis')
col1, col2 = st.columns(2)
with col1:
    st.write("Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=titanic_df, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("Survival by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', hue='Sex', data=titanic_df, ax=ax)
    st.pyplot(fig)

# Preprocessing
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)
titanic_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

le = LabelEncoder()
titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])
titanic_df['Embarked'] = le.fit_transform(titanic_df['Embarked'])

X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results
st.subheader('Model Evaluation')
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.subheader('Confusion Matrix')
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader('Feature Importance')
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.coef_[0]})
feature_importance = feature_importance.sort_values('importance', ascending=False)
fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
st.pyplot(fig)

else:
    st.info('☝️ Upload a CSV file')
