import pandas as pd
import numpy as np
from challenge_functions.baseline_algorithm.functions import string_to_array
import gc


def explode(df_in, col_expl):
    """Explode column col_expl of array type into multiple rows."""
    # Col_expl - "impressions"
    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    gc.collect()

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    gc.collect()

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    df_out.loc[:, col_expl] = df_out[col_expl].apply(str)

    return df_out


def get_features():
    df = pd.read_csv('./data/item_metadata.csv')

    df['properties'] = df['properties'].str.replace(' ', '_')
    gc.collect()
    df_out = explode(df, 'properties')
    gc.collect()

    # len(df_out.properties.unique()) # 157 unikalnych cech
    gc.collect()

    print("Unique properties number: ", len(df_out.properties.unique()))

    print("Number of occuring")
    print(df_out.properties.value_counts())
    df_out.properties.value_counts().to_csv('./data/features_count.csv')


def extract_features():
    df = pd.read_csv('./data/item_metadata.csv')
    df['properties'] = df['properties'].str.replace(' ', '_')
    gc.collect()

    # Wybrane cechy
    features = ['Car_Park', 'Shower', 'Pet_Friendly', 'Reception_(24/7)']
    rating = ['Satisfactory_Rating', 'Good_Rating', 'Very_Good_Rating']
    wifi = ['WiFi_(Rooms)', 'WiFi_(Public_Areas)', 'Free_WiFi_(Combined)', 'Free_WiFi_(Rooms)', 'Free_WiFi_(Public_Areas)']

    for f in features:
        df[f] = np.where(df.properties.str.contains(f, regex=False), 1, 0)

    df['Positive_Rating'] = np.where(df.properties.str.contains(rating, regex=False), 1, 0)
    df['WiFi'] = np.where(df.properties.str.contains(wifi, regex=False), 1, 0)

    df = df.drop('properties', axis=1)
    print(df.head())
    df.to_csv('./processed/tested_features.csv', index=False)


def other_features():
    df = pd.read_csv('./data/item_metadata.csv')
    df['properties'] = df['properties'].str.replace(' ', '_')
    gc.collect()

    df['3_Star'] = np.where(df.properties.str.contains('|3_Star|', regex=False), 1, 0)
    df['Reception_(24/7)'] = np.where(df.properties.str.contains('Reception_(24/7)', regex=False), 1, 0)
    df['Satisfactory_Rating'] = np.where(df.properties.str.contains('Satisfactory_Rating', regex=False), 1, 0)
    df['Car_Park'] = np.where(df.properties.str.contains('Car_Park', regex=False), 1, 0)
    df['Very_Good_Rating'] = np.where(df.properties.str.contains('Very_Good_Rating', regex=False), 1, 0)
    df = df.drop('properties', axis=1)
    df.to_csv('./processed/5_features.csv', index=False)


if __name__ == '__main__':
    extract_features()