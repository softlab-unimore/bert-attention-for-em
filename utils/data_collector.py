import pandas as pd
import os
import wget
import pathlib
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
DM_USE_CASES = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar",
                "Structured_DBLP-ACM", "Structured_Amazon-Google",
                "Structured_Walmart-Amazon", "Structured_Beer",
                "Structured_iTunes-Amazon", "Textual_Abt-Buy",
                "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
WDC_USE_CASES = ['Xlarge_Computers', 'Xlarge_Cameras', 'Xlarge_Shoes', 'Xlarge_Watches',
                 'Large_Computers', 'Large_Cameras', 'Large_Shoes', 'Large_Watches']


class DataCollector(object):
    def __init__(self, data_dir: str = 'data'):

        assert isinstance(data_dir, str), "Wrong data directory type."

        self.data_dir = os.path.join(PROJECT_DIR, data_dir)

    def _get_complete_dataset(self, dataset, use_case_data_dir):
        """
        Expand the dataset with the records from source A and source B.
        """

        dataset_path = os.path.join(use_case_data_dir, dataset)
        ds = pd.read_csv(dataset_path)

        tableA_path = os.path.join(use_case_data_dir, "tableA.csv")
        ds_a = pd.read_csv(tableA_path)

        tableB_path = os.path.join(use_case_data_dir, "tableB.csv")
        ds_b = pd.read_csv(tableB_path)

        assert 'ltable_id' in ds
        assert 'rtable_id' in ds
        assert 'id' in ds_b
        assert 'id' in ds_a

        ds_a = ds_a.add_prefix('left_')
        ds_b = ds_b.add_prefix('right_')

        ds = pd.merge(ds, ds_a, how='inner', left_on='ltable_id', right_on='left_id', suffixes=(False, False))
        ds = pd.merge(ds, ds_b, how='inner', left_on='rtable_id', right_on='right_id', suffixes=(False, False))

        # ds.drop(["ltable_id", "rtable_id", "left_id", "right_id"], axis=1, inplace=True)
        ds.drop(["ltable_id", "rtable_id"], axis=1, inplace=True)

        return ds

    def _save_complete_dataset(self, dataset, use_case_data_dir):
        """
        Expand the integrated dataset.
        """

        ds = self._get_complete_dataset(dataset, use_case_data_dir)

        out_file_name = os.path.join(use_case_data_dir, dataset)
        ds.to_csv(out_file_name, index=False)

    def _download_data(self, use_case, use_case_data_dir):
        """
        Download the datasets associated to the provided DeepMatcher use case.
        """

        use_case = use_case.replace(os.sep, "/")

        base_url = "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/{}/exp_data".format(use_case)
        tableA = "{}/tableA.csv".format(base_url)
        tableB = "{}/tableB.csv".format(base_url)
        train = "{}/train.csv".format(base_url)
        test = "{}/test.csv".format(base_url)
        valid = "{}/valid.csv".format(base_url)

        wget.download(tableA, out=use_case_data_dir)
        wget.download(tableB, out=use_case_data_dir)
        wget.download(train, out=use_case_data_dir)
        wget.download(test, out=use_case_data_dir)
        wget.download(valid, out=use_case_data_dir)

        # extend datasets
        self._save_complete_dataset("train.csv", use_case_data_dir)
        self._save_complete_dataset("test.csv", use_case_data_dir)
        self._save_complete_dataset("valid.csv", use_case_data_dir)

    def get_data(self, use_case):

        assert isinstance(use_case, str), "Wrong use case type."
        assert use_case in DM_USE_CASES, "Wrong use case name."
        use_case = use_case.replace("_", os.sep)

        print(f"USE CASE: {use_case}")

        use_case_data_dir = os.path.join(self.data_dir, use_case)

        # check data existence
        if not (os.path.exists(os.path.join(use_case_data_dir, "tableA.csv")) and
                os.path.exists(os.path.join(use_case_data_dir, "tableB.csv")) and
                os.path.exists(os.path.join(use_case_data_dir, "train.csv")) and
                os.path.exists(os.path.join(use_case_data_dir, "valid.csv")) and
                os.path.exists(os.path.join(use_case_data_dir, "test.csv"))):

            print("Starting downloading the data...")
            for file in ["tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv"]:
                f = os.path.join(use_case_data_dir, file)
                if os.path.exists(f):
                    os.remove(f)

            pathlib.Path(use_case_data_dir).mkdir(parents=True, exist_ok=True)

            self._download_data(use_case, use_case_data_dir)

        else:
            print("Data already downloaded.")

        return use_case_data_dir

    def get_dm_benchmark(self):
        for use_case in DM_USE_CASES:
            self.get_data(use_case)


class DataCollectorWDC:
    def __init__(self, data_dir: str = 'data'):

        assert isinstance(data_dir, str), "Wrong data directory type."

        self.data_dir = os.path.join(PROJECT_DIR, data_dir)

    def get_data(self, use_case):

        assert isinstance(use_case, str), "Wrong use case type."
        assert use_case in WDC_USE_CASES, "Wrong use case name."
        use_case = use_case.replace("_", os.sep)

        print(f"USE CASE: {use_case}")

        use_case_data_dir = os.path.join(self.data_dir, use_case)

        # check data existence
        if not (os.path.exists(os.path.join(use_case_data_dir, "train.csv")) and
                os.path.exists(os.path.join(use_case_data_dir, "valid.csv")) and
                os.path.exists(os.path.join(use_case_data_dir, "test.csv"))):
            raise ValueError("Data not found!")

        else:
            print("Data already downloaded.")

        return use_case_data_dir
