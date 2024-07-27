python
import pandas as pd

# assuming your DataFrame is named df

def extract_data_for_bar_chart(df):
    genre_popularity_data = df.groupby('genre')['popularity'].mean().reset_index()
    return genre_popularity_data.to_csv(index=False, columns=['genre', 'popularity'])

# call the function
genre_popularity_data = extract_data_for_bar_chart(your_df)
print(genre_popularity_data)