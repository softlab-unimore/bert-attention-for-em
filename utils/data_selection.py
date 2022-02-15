from core.data_models.em_dataset import EMDataset
import pandas as pd


class Sampler(object):
    """
    This class implements some techniques for sampling rows from an EM dataset.
    """

    def __init__(self, dataset: EMDataset, permute: bool = False):

        assert isinstance(dataset, EMDataset), "Wrong data type for parameter 'dataset'."
        assert isinstance(permute, bool), "Wrong data type for parameter 'permute'."

        self.dataset = dataset
        self.data = self.dataset.get_complete_data()
        self.dataset_params = self.dataset.get_params()
        self.permute = permute

    def _get_data_by_label(self, label_val: int, size: int = None, seed: int = 42):

        assert isinstance(label_val, int), "Wrong data type for parameter 'label_val'."
        if size is not None:
            assert isinstance(size, int), "Wrong data type for parameter 'size'."
        assert isinstance(seed, int), "Wrong data type for parameter 'seed'."

        label_col = self.dataset_params['label_col']

        assert label_val in list(self.data[label_col].unique()), "Wrong value for parameter 'label_val'."

        out_data = self.data[self.data[label_col] == label_val]

        if size is not None:
            if size > len(out_data):
                old_size = size
                size = len(out_data)
                print(f"No enough data for size {old_size}. Taken {size} records.")

            out_data = out_data.sample(size, random_state=seed)

        return out_data

    def _create_dataset(self, data: pd.DataFrame, params: dict):

        assert isinstance(data, pd.DataFrame), "Wrong data type for parameter 'data'."
        assert isinstance(params, dict), "Wrong data type for parameter 'params'."
        param_names = ["model_name", "label_col", "left_prefix", "right_prefix", "max_len", "verbose", "tokenization"]
        assert all([p in params for p in param_names]), "Missing some parameters from 'params'."

        model_name = params["model_name"]
        label_col = params["label_col"]
        left_prefix = params["left_prefix"]
        right_prefix = params["right_prefix"]
        max_len = params["max_len"]
        verbose = params["verbose"]
        tokenization = params["tokenization"]
        return_offset = params["return_offset"]

        return EMDataset(data, model_name, tokenization=tokenization, label_col=label_col, left_prefix=left_prefix,
                         right_prefix=right_prefix, max_len=max_len, verbose=verbose, permute=self.permute,
                         return_offset=return_offset)

    def get_match_data(self, size: int = None, seed: int = 42):

        match_data = self._get_data_by_label(1, size, seed)

        dataset_params = self.dataset_params.copy()
        dataset_params["verbose"] = True

        return self._create_dataset(match_data, dataset_params)

    def get_non_match_data(self, size: int = None, seed: int = 42):

        non_match_data = self._get_data_by_label(0, size, seed)

        dataset_params = self.dataset_params.copy()
        dataset_params["verbose"] = True

        return self._create_dataset(non_match_data, dataset_params)

    def get_balanced_data(self, size: int = None, seeds: list = [42, 42]):

        assert isinstance(seeds, list), "Wrong data type for parameter 'seeds'."
        assert len(seeds) == 2, "Only two classes are supported."

        match_seed = seeds[0]
        non_match_seed = seeds[1]

        match_data = self._get_data_by_label(1, size, match_seed)
        if size is not None:
            # if no enough match data is available, limit also the non-match data in order to create a balanced sample
            if len(match_data) < size:
                size = len(match_data)
            non_match_data = self._get_data_by_label(0, size, non_match_seed)
        else:
            non_match_data = self._get_data_by_label(0, len(match_data), non_match_seed)

        out_data = pd.concat([match_data, non_match_data])

        dataset_params = self.dataset_params.copy()
        dataset_params["verbose"] = True
        # dataset_params["categories"] = ([1] * len(match_data)) + ([0] * len(match_data))

        return self._create_dataset(out_data, dataset_params)
