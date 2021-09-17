
attr_attn_patterns = {
    "script": "attention_pattern_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'use_case': "Structured_Fodors-Zagats",                                 # this will ignored
        'data_type': 'train',
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # 'sent_pair', 'attr_pair'
        'size': None,
        'fine_tune_method': None,                                               # None, 'simple'
        'extractor': {
            'attn_extractor': 'attr_extractor',
            'attn_extr_params': {'special_tokens': True, 'agg_metric': 'max'},  # 'max', 'mean'
        },
        'tester': {
            'tester': 'attr_tester',
            'tester_params': {'ignore_special': True}
        },
    },
    "experiment": 'pattern',
    "analysis_target": 'benchmark',
    "analysis_type": 'multi',
    "sub_experiment": None,
    "comparison": None,
    "plot_params": ['match_attr_attn_over_mean'],
    "agg_fns": None,
    "target_agg_result_ids": None,
    "categories": ['all'],
    "comparison_param": None,
    "small_plot": None
}

agg_maa_pattern_pt_vs_ft = {
    "script": "attention_pattern_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'use_case': "Structured_Fodors-Zagats",                                 # this will ignored
        'data_type': 'train',
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # 'sent_pair', 'attr_pair'
        'size': None,
        'fine_tune_method': None,                                               # this will ignored
        'extractor': {
            'attn_extractor': 'attr_extractor',
            'attn_extr_params': {'special_tokens': True, 'agg_metric': 'max'},  # 'mean', 'max'
        },
        'tester': {
            'tester': 'attr_tester',
            'tester_params': {'ignore_special': True}
        },
    },
    "experiment": 'pattern',
    "analysis_target": 'benchmark',
    "analysis_type": 'comparison',
    "sub_experiment": None,
    "comparison": None,
    "plot_params": ['match_attr_attn_over_mean'],
    "agg_fns": ['row_mean', 'row_std'],
    "target_agg_result_ids": ['match_attr_attn_loc'],
    "categories": ['all'],
    "comparison_param": 'fine_tune_method',
    "small_plot": None
}

maa_pattern_comparison_by_layer = {
    "script": "attention_pattern_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'use_case': "Structured_Fodors-Zagats",                                 # this will ignored
        'data_type': 'train',
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # this will ignored
        'size': None,
        'fine_tune_method': None,                                               # this will ignored
        'extractor': {
            'attn_extractor': 'attr_extractor',
            'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
        },
        'tester': {
            'tester': 'attr_pattern_tester',
            'tester_params': {'ignore_special': True}
        },
    },
    "experiment": 'pattern_freq',
    "analysis_target": None,
    "analysis_type": 'comparison',
    "sub_experiment": 'match_freq_by_layer',
    "comparison": 'tune_tok',
    "plot_params": None,
    "agg_fns": None,
    "target_agg_result_ids": None,
    "categories": None,
    "comparison_param": None,
    "small_plot": False
}

maa_pattern_comparison_by_layer_small = maa_pattern_comparison_by_layer.copy()
maa_pattern_comparison_by_layer_small["use_cases"] = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar",
                                                      "Structured_DBLP-ACM", "Structured_Amazon-Google"]
maa_pattern_comparison_by_layer_small["comparison"] = "tune"
maa_pattern_comparison_by_layer_small["small_plot"] = True

pattern_freq_comparison = {
    "script": "attention_pattern_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'use_case': "Structured_Fodors-Zagats",                                 # this will ignored
        'data_type': 'train',
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # this will ignored
        'size': None,
        'fine_tune_method': None,                                               # this will ignored
        'extractor': {
            'attn_extractor': 'attr_extractor',
            'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
        },
        'tester': {
            'tester': 'attr_pattern_tester',
            'tester_params': {'ignore_special': True}
        },
    },
    "experiment": 'pattern_freq',
    "analysis_target": None,
    "analysis_type": 'comparison',
    "sub_experiment": 'all_freq',
    "comparison": 'tune_tok',
    "plot_params": None,
    "agg_fns": None,
    "target_agg_result_ids": None,
    "categories": None,
    "comparison_param": None,
    "small_plot": False
}

attn_to_pos = {
    "script": "attention_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'data_type': 'train',
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # 'sent_pair', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    },
    "sampler_conf": {
        'size': None,
        'target_class': 'both',
        'seeds': [42, 42],
    },
    "fine_tune": None,                                                          # None, 'simple'
    "attn_params": {
        'attn_extractor': 'word_extractor',
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    },
    "experiment": 'topk_attn_word',
    "sub_experiment": 'simple',
    "precomputed": True,
    "multi": False,
    "target_methods": None,
    "pos_cat": None,
    "agg_plot": False,
    "agg_dim": None,
    "small_plot": False,
    "comparison": None
}

attn_to_pos_small = attn_to_pos.copy()
attn_to_pos_small["use_cases"] = ["Structured_Fodors-Zagats", "Structured_Walmart-Amazon", "Structured_iTunes-Amazon",
                                  "Textual_Abt-Buy"]
attn_to_pos_small["small_plot"] = True

attn_to_text_pos_comp_uc = {
    "script": "attention_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'data_type': 'train',
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # this will ignored
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    },
    "sampler_conf": {
        'size': None,
        'target_class': 'both',
        'seeds': [42, 42],
    },
    "fine_tune": None,                                                          # this will ignored
    "attn_params": {
        'attn_extractor': 'word_extractor',
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    },
    "experiment": 'topk_attn_word',
    "sub_experiment": 'comparison',
    "precomputed": True,
    "multi": True,
    "target_methods": ['finetune_sentpair', 'finetune_attrpair', 'pretrain_sentpair', 'pretrain_attrpair'],
    "pos_cat": "TEXT",
    "agg_plot": True,
    "agg_dim": 'layer',
    "small_plot": False,
    "comparison": None
}

attn_to_text_pos_comp_layer = attn_to_text_pos_comp_uc.copy()
attn_to_text_pos_comp_layer["agg_dim"] = 'use_case'

attn_to_text_pos_comp_uc_small = attn_to_text_pos_comp_uc.copy()
attn_to_text_pos_comp_uc_small["target_methods"] = ['pretrain_sentpair', 'finetune_sentpair']
attn_to_text_pos_comp_uc_small["small_plot"] = True

attn_to_text_pos_comp_layer_small = attn_to_text_pos_comp_layer.copy()
attn_to_text_pos_comp_layer_small["target_methods"] = ['pretrain_sentpair', 'finetune_sentpair']
attn_to_text_pos_comp_layer_small["small_plot"] = True

attn_to_syntactic_comp_uc = {
    "script": "attention_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'data_type': 'train',
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # this will ignored
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    },
    "sampler_conf": {
        'size': None,
        'target_class': 'both',
        'seeds': [42, 42],
    },
    "fine_tune": None,                                                          # this will ignored
    "attn_params": {
        'attn_extractor': 'word_extractor',
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    },
    "experiment": 'topk_word_attn_by_attr_similarity',
    "sub_experiment": 'comparison',
    "precomputed": True,
    "multi": True,
    "target_methods": ['finetune_sentpair', 'finetune_attrpair', 'pretrain_sentpair', 'pretrain_attrpair'],
    "pos_cat": None,
    "agg_plot": True,
    "agg_dim": 'layer',
    "small_plot": False,
    "comparison": None
}

attn_to_syntactic_comp_layer = attn_to_syntactic_comp_uc.copy()
attn_to_syntactic_comp_layer['agg_dim'] = 'use_case'

attn_to_syntactic_comp_uc_small = attn_to_syntactic_comp_uc.copy()
attn_to_syntactic_comp_uc_small['small_plot'] = True
attn_to_syntactic_comp_uc_small["target_methods"] = ['pretrain_sentpair', 'finetune_sentpair']

attn_to_syntactic_comp_layer_small = attn_to_syntactic_comp_layer.copy()
attn_to_syntactic_comp_layer_small['small_plot'] = True
attn_to_syntactic_comp_layer_small["target_methods"] = ['pretrain_sentpair', 'finetune_sentpair']

cls_to_attr_attn_pt_vs_ft = {
    "script": "attention_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'data_type': 'train',
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # this will ignored
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    },
    "sampler_conf": {
        'size': None,
        'target_class': 'both',
        'seeds': [42, 42],
    },
    "fine_tune": None,                                                          # this will ignored
    "attn_params": {
        'attn_extractor': 'attr_extractor',
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'max'},
    },
    "experiment": 'attr_to_cls',
    "sub_experiment": 'comparison',
    "precomputed": None,
    "multi": None,
    "target_methods": None,
    "pos_cat": None,
    "agg_plot": None,
    "agg_dim": None,
    "small_plot": False,
    "comparison": "tune"
}

cls_to_attr_attn_pt_vs_ft_small = cls_to_attr_attn_pt_vs_ft.copy()
cls_to_attr_attn_pt_vs_ft_small["use_cases"] = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar",
                                       "Structured_DBLP-ACM", "Dirty_DBLP-ACM"]
cls_to_attr_attn_pt_vs_ft_small["small_plot"] = True

cls_to_attr_attn_ft_match_vs_non_match = cls_to_attr_attn_pt_vs_ft.copy()
cls_to_attr_attn_ft_match_vs_non_match["sub_experiment"] = "simple"
cls_to_attr_attn_ft_match_vs_non_match["fine_tune"] = "simple"

cls_to_attr_attn_ft_match_vs_non_match_small = cls_to_attr_attn_ft_match_vs_non_match.copy()
cls_to_attr_attn_ft_match_vs_non_match_small["use_cases"] = ["Structured_Fodors-Zagats",
                                                             "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                                                             "Dirty_DBLP-ACM"]
cls_to_attr_attn_ft_match_vs_non_match_small["small_plot"] = True

e2e_cmp = {
    "script": "attention_test.py",
    "use_cases": ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
    "conf": {
        'data_type': 'train',
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                                                     # this will ignored
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    },
    "sampler_conf": {
        'size': None,
        'target_class': 'both',
        'seeds': [42, 42],
    },
    "fine_tune": None,                                                          # this will ignored
    "attn_params": {
        'attn_extractor': 'token_extractor',
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    },
    "experiment": 'entity_to_entity',
    "sub_experiment": 'comparison',
    "precomputed": None,
    "multi": None,
    "target_methods": None,
    "pos_cat": None,
    "agg_plot": None,
    "agg_dim": None,
    "small_plot": False,
    "comparison": "tune_tok"
}

e2e_cmp_small = e2e_cmp.copy()
e2e_cmp_small["use_cases"] = ["Structured_Amazon-Google", "Structured_Beer", "Textual_Abt-Buy", "Dirty_Walmart-Amazon"]
e2e_cmp_small["small_plot"] = True
e2e_cmp_small["comparison"] = "tune"

e2e_ft_match_vs_non_match = e2e_cmp_small.copy()
e2e_ft_match_vs_non_match["fine_tune"] = "simple"
e2e_ft_match_vs_non_match["comparison"] = "tok"

e2e_ft_match_vs_non_match_small = e2e_ft_match_vs_non_match.copy()
e2e_ft_match_vs_non_match_small["sub_experiment"] = 'simple'
