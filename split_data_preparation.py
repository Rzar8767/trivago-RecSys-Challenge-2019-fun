import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def save_small_set(df):
    shrink_set(df, "small_set", int(len(df) * 0.01))


def save_medium_set(df):
    shrink_set(df, "medium_set", int(len(df) * 0.2))


def save_large_set(df):
    shrink_set(df, "large_set", int(len(df) * 0.4), )


# Source: https://github.com/keyblade95/recsys2019/blob/master/preprocess.py
def shrink_set(df, name, maximum_rows=1000000):
    """
    Saves a dataframe from the original dataset containing a maximum number of rows. The actual total rows
    extracted may vary in order to avoid breaking the last session.
    :param name: string
    :param df: dataframe
    :param maximum_rows:
    :return: nothing
    """
    if len(df) < maximum_rows:
        df.to_csv(f"./processed/{name}.csv", sep=",", index=False)
    # get the last row
    last_row = df.iloc[[maximum_rows]]
    last_session_id = last_row.session_id.values[0]

    # OPTIMIZATION: last_user_id = last_row.user_id.values[0]

    # slice the dataframe from the target row on
    temp_df = df.iloc[maximum_rows:]
    # get the number of remaining interactions of the last session
    # OPTIMIZATION: remaining_rows = temp_df[(temp_df.session_id == last_session_id) & (temp_df.user_id == last_user_id)].shape[0]
    remaining_rows = temp_df[temp_df.session_id == last_session_id].shape[0]
    # slice from the first row to the final index
    df.iloc[0:maximum_rows + remaining_rows].to_csv(f"./processed/{name}.csv", sep=",", index=False)


# Source: https://github.com/keyblade95/recsys2019/blob/master/preprocess.py
def reset_step_for_duplicated_sessions(df):
    """ Reset the step for some bugged session in which the step restart from 1 in some random interaction """
    res_df = df.copy()
    # find the sessions in which the step restarts at some point
    df_dup = df[["session_id", "user_id", "step"]]
    df_dup = df_dup[df_dup["step"] == 1]
    df_dup = df_dup.groupby(['user_id', 'session_id', 'step']).size() \
        .sort_values(ascending=False) \
        .reset_index(name='count')
    df_dup = df_dup[df_dup["count"] > 1]
    df_dup = df_dup[['user_id', 'session_id']]

    # reset the steps for the duplicated-steps sessions
    for _, row in tqdm(df_dup.iterrows()):
        mask = (df.user_id == row.user_id) & (df.session_id == row.session_id)
        sess_length = sum(mask * 1)
        res_df.loc[mask, 'step'] = np.arange(1, sess_length + 1, dtype='int')

    return res_df


def load_small_set():
    return pd.read_csv('./processed/small_set.csv')


def load_medium_set():
    return pd.read_csv('./processed/medium_set.csv')


def load_large_set():
    return pd.read_csv('./processed/large_set.csv')


# Split based on % of the duration
# Samples that are sliced will be moved to train set
def split_dataset(df, percent, name):
    duration = df.timestamp.max() - df.timestamp.min()
    split_time = df.timestamp.min() + int(duration * percent)
    df_train = df[df.timestamp <= split_time]
    df_test = df[df.timestamp > split_time]
    print("Last of train set:")
    print_dataset(df_train.tail())
    print("First of test set:")
    print_dataset(df_test.head())
    print("Length of train set: ", len(df_train))
    print("Length of test set: ", len(df_test))

    # Move interrupted sessions into the train set

    # Get sessions that are in test and train set and find the overlap
    only_sessions_train = df_train.iloc[:, 1:2].copy()
    only_sessions_test = df_test.iloc[:, 1:2].copy()
    overlap = pd.merge(only_sessions_train, only_sessions_test, how='inner')

    # Locate all the sessions
    print("Sessions in train set:", len(df_train.session_id.unique()))
    print("Sessions in test set:", len(df_test.session_id.unique()))
    print("Sessions cut in half:\n", overlap.session_id.unique())

    overlap_unique = overlap.session_id.unique()

    overlap_from_test = df_test[df.session_id.isin(overlap_unique)]
    df_train = pd.merge(df_train, overlap_from_test, how='outer')
    df_test = df_test[~df.session_id.isin(overlap_unique)]
    ground_truth = df_test.copy()

    # Test if sessions still overlap in the dataframes
    is_session_overlap(df_train, df_test)

    # Hide last checkouts
    test = df_test.head(15)
    hide_last_checkouts(test)
    print(test)

    df_train.to_csv(f"./processed/train_set_{name}.csv", sep=",", index=False)
    df_test.to_csv(f"./processed/test_set_{name}.csv", sep=",", index=False)
    ground_truth.to_csv(f"./processed/ground_truth_{name}.csv", sep=",", index=False)


# Source: https://github.com/keyblade95/recsys2019/blob/master/preprocess.py
def is_session_overlap(df1, df2):
    df1_sessions = df1.iloc[:, 1:2].copy()
    df2_sessions = df2.iloc[:, 1:2].copy()
    overlap = pd.merge(df1_sessions, df2_sessions, how='inner')
    assert len(overlap) == 0, "is_session_overlap: Datasets overlap!"


def hide_last_checkouts(df_test):
    groups = df_test[df_test['action_type'] == 'clickout item'].groupby('user_id', as_index=False)
    remove_reference_tuples = groups.apply(lambda x: x.sort_values(by=['timestamp'], ascending=True).tail(1))

    for index, row in tqdm(remove_reference_tuples.iterrows()):
        if int(row['reference']) not in list(map(int, row['impressions'].split('|'))):
            remove_reference_tuples.drop(index, inplace=True)

    for e in tqdm(remove_reference_tuples.index.tolist()):
        df_test.at[e[1], 'reference'] = np.nan


def print_dataset(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 240)
    print(df.iloc[:, 0:7])


if __name__ == '__main__':
    recreate_sets = False
    full_set = True
    if recreate_sets:
        df = pd.read_csv('./data/train.csv')
        df = reset_step_for_duplicated_sessions(df)
        print(len(df))
        save_small_set(df)
        save_medium_set(df)
        save_large_set(df)
    if full_set:
        full = pd.read_csv('./data/train.csv')
        full = reset_step_for_duplicated_sessions(full)
        print(len(full))
        split_dataset(full, 0.8, "full")
    else:
        medium = load_medium_set()
        print(len(medium))
        split_dataset(medium, 0.8, "medium")
