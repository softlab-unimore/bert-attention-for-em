
class ConfCreator(object):

    def __init__(self):
        self.conf_template = {
            'use_case': ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                         "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                         "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                         "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
            'data_type': ['train', 'test', 'valid'],
            'permute': [True, False],
            'model_name': ['bert-base-uncased'],
            'tok': ['attr_pair', 'sent_pair'],
            'size': [None],
            'fine_tune_method': [None, 'simple'],   # ['advanced', 'simple', None],
            'extractor': {
                'attn_extractor': ['attr_extractor', 'word_extractor'],
                'attn_extr_params': ['special_tokens', 'agg_metric'],
            },
            'tester': {
                'tester': ['attr_tester', 'attr_pattern_tester'],
                'tester_params': ['ignore_special']
            },
        }
        self.use_case_map = {
            "Structured_Fodors-Zagats": 'S-FZ',
            "Structured_DBLP-GoogleScholar": 'S-DG',
            "Structured_DBLP-ACM": 'S-DA',
            "Structured_Amazon-Google": 'S-AG',
            "Structured_Walmart-Amazon": 'S-WA',
            "Structured_Beer": 'S-BR',
            "Structured_iTunes-Amazon": 'S-IA',
            "Textual_Abt-Buy": 'T-AB',
            "Dirty_iTunes-Amazon": 'D-IA',
            "Dirty_DBLP-ACM": 'D-DA',
            "Dirty_DBLP-GoogleScholar": 'D-DG',
            "Dirty_Walmart-Amazon": 'D-WA'
        }

    def validate_conf(self, conf: dict):
        assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
        assert all([p in self.conf_template for p in conf]), "Wrong data format for parameter 'conf'."

        err_msg = "Wrong data format for parameter 'conf'."
        for (k, v) in conf.items():
            if k in ['extractor', 'tester']:
                assert isinstance(v, dict), err_msg
                assert all([p in self.conf_template[k] for p in v.keys()]), err_msg
                for subk, subv in v.items():
                    if isinstance(subv, dict):
                        assert all([p in self.conf_template[k][subk] for p in subv.keys()]), err_msg
                    else:
                        assert subv in self.conf_template[k][subk], err_msg
            else:
                assert v in self.conf_template[k] or v == self.conf_template[k], err_msg

        return conf

    def get_confs(self, conf: dict, params: list):
        conf = self.validate_conf(conf)
        assert isinstance(params, list), "Wrong data type for parameter 'params'."
        assert all([isinstance(p, str) for p in params]), "Wrong data type for parameter 'params'."
        assert all([p in self.conf_template for p in params]), "Wrong value for parameter 'params'."

        confs = []
        for param in params:
            for val in self.conf_template[param]:
                out_conf = conf.copy()
                out_conf[param] = val
                confs.append(out_conf)

        return confs

    def get_param_values(self, param: str):
        assert isinstance(param, str), "Wrong data type for parameter 'param'."
        assert param in self.conf_template, "Wrong value for parameter 'param'."

        return self.conf_template[param]
