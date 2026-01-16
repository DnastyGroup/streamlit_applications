import streamlit as st
import pandas as pd

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

else:
    st.info('☝️ Upload a CSV file')
