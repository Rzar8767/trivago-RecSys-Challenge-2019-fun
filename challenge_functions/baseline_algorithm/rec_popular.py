from pathlib import Path

import click
import pandas as pd
import gc

from . import functions as f

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


#@click.command()
#@click.option('--data-path', default=None, help='Directory for the CSV files')
def main(data_path, train_path, test_path, subm_path):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = train_path
    test_csv = test_path
    subm_csv = subm_path

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)
    print(f"Reading tested_features.csv")
    df_ft = pd.read_csv('./processed/tested_features.csv')

    gc.collect()

    print("Get popular items...")
    #df_popular = f.get_popularity(df_train)
    df_popular = f.get_popularity_and_information(df_train, df_ft)
    gc.collect()

    print("Identify target rows...")
    df_target = f.get_submission_target(df_test)

    print("Get recommendations...")
    df_expl = f.explode(df_target, "impressions")
    df_out = f.calc_recommendation(df_expl, df_popular)

    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()
