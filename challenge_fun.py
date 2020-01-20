from challenge_functions.score_submission import score_subm as ss
from challenge_functions.verify_submission import verify_subm as vs
from challenge_functions.baseline_algorithm import rec_popular as rp

file = 'full'

subm_csv = f'./processed/submission_popular_{file}.csv'
gt_csv = f'./processed/ground_truth_{file}.csv'
test_csv = f'./processed/test_set_{file}.csv'
train_csv = f'./processed/train_set_{file}.csv'
data_path = '.'


def baseline():
    rp.main(data_path, train_csv, test_csv, subm_csv)


def verify():
    vs.main(data_path, subm_csv, test_csv)


def score():
    ss.main(data_path, subm_csv, gt_csv)


if __name__ == '__main__':
    baseline()
    verify()
    score()
