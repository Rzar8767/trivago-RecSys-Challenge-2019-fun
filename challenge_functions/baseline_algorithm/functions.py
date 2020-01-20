import math
import pandas as pd
import numpy as np
import gc

GR_COLS = ["user_id", "session_id", "timestamp", "step"]


def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out


def get_popularity(df):
    """Get number of clicks that each item received in the df."""

    mask = df["action_type"] == "clickout item"
    df_clicks = df[mask]
    df_item_clicks = (
        df_clicks
        .groupby("reference")
        .size()
        .reset_index(name="n_clicks")
        .transform(lambda x: x.astype(int))
    )

    return df_item_clicks


def get_popularity_and_information(df, df_meta):
    """Get number of clicks that each item received in the df."""

    mask = df["action_type"] == "clickout item"
    df_clicks = df[mask]
    df_item_clicks = (
        df_clicks
        .groupby("reference")
        .size()
        .reset_index(name="n_clicks")
        .transform(lambda x: x.astype(int))
    )
    gc.collect()
    # df_item_clicks -> reference & n_clicks

    df_expl = explode(df[mask], "impressions")
    df_expl['reference'] = df_expl['reference'].astype(int)
    # df_expl -> user_id, session_id, timestamp, step, action_type, reference, platform, city,
    # device, current_filters, prices, impressions
    gc.collect()

    n_impressions = df_expl.groupby("impressions").size().to_frame().reset_index()
    n_impressions.columns = ["reference", "n_impressions"]

    gc.collect()
#----------------------------

    df_out = pd.merge(n_impressions, df_item_clicks, on="reference", how='left')
    df_out = df_out.fillna(0)

    # Calculate click through rate calculation
    df_out['ctr'] = df_out['n_clicks'] / df_out['n_impressions']

    df_out['ctr'] = np.where((df_out['ctr'] >= 0.85) & (df_out['ctr'] < 1), df_out['ctr'], 0)

    gc.collect()

    df_meta = df_meta.rename(columns={'item_id': 'reference'})
    df_out = pd.merge(df_out, df_meta, on='reference', how='right')
    df_out = df_out.fillna(0)

    print(df_out.head())

    gc.collect()

    df_out['meta_info'] = df_out.iloc[:, 4:].sum(axis=1)/10
    print(df_out['meta_info'].head())

    # When ctr is between <0.85;1) base popularity on ctr + info based on whether hotels have the desired features
    df_out['popularity'] = np.where((df_out['ctr'] >= 0.85) & (df_out['ctr'] < 1), df_out['ctr']+df_out['meta_info'], 0)

    df_out = df_out[['reference', 'popularity']]

    return df_out


def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


def explode(df_in, col_expl):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    df_out.loc[:, col_expl] = df_out[col_expl].apply(int)

    return df_out


def group_concat(df, gr_cols, col_concat):
    """Concatenate multiple rows into one."""

    df_out = (
        df
        .groupby(gr_cols)[col_concat]
        .apply(lambda x: ' '.join(x))
        .to_frame()
        .reset_index()
    )

    return df_out


def calc_recommendation(df_expl, df_pop):
    """Calculate recommendations based on popularity of items.

    The final data frame will have an impression list sorted according to the number of clicks per item in a reference data frame.

    :param df_expl: Data frame with exploded impression list
    :param df_pop: Data frame with items and number of clicks
    :return: Data frame with sorted impression list according to popularity in df_pop
    """

    df_expl_clicks = (
        df_expl[GR_COLS + ["impressions"]]
        .merge(df_pop,
               left_on="impressions",
               right_on="reference",
               how="left")
    )

    df_out = (
        df_expl_clicks
        .assign(impressions=lambda x: x["impressions"].apply(str))
        #.sort_values(GR_COLS + ["n_clicks"],
        #             ascending=[True, True, True, True, False])
        .sort_values(GR_COLS + ["popularity"],
                     ascending=[True, True, True, True, False])
    )

    df_out = group_concat(df_out, GR_COLS, "impressions")
    df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)

    return df_out
