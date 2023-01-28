import pandas as pd

from core.data_models.em_dataset import EMDataset
from utils.data_selection import Sampler


class GeneralPerturbation:
    pass


class RelevanceAttributePerturbation(GeneralPerturbation):
    def __init__(self, dataset: EMDataset):
        self.sampler = Sampler(dataset)
        self.data = self.sampler._get_data_by_label(1)
        self.dataset_params = self.sampler.dataset_params.copy()
        self.dataset_params["verbose"] = False

    def perturb_row(self, row: pd.Series, context: pd.DataFrame):

        row_perturbations = [row, row]
        return row_perturbations

    def perturb_dataset(self, data: pd.DataFrame):

        out_data = []
        for ix, row in data.iterrows():
            out_rows = self.perturb_row(row, data)
            df_from_row = pd.concat(out_rows)
            out_data.append(df_from_row)

        return pd.concat(out_data)

    def apply_perturbation(self):
        data = self.data.copy()
        out_data = self.perturb_dataset(data)
        return self.sampler._create_dataset(out_data, self.dataset_params)
