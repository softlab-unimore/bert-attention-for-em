import os.path
import pandas as pd
from utils.cluster_matching_records import convert_matching_pairs_to_integrated_dataset


def split_train_valid(train_data_path, valid_data_path):
    # Load train sets and valid pairs
    train = pd.read_json(train_data_path, compression='gzip', lines=True)
    valid_pairs = pd.read_csv(valid_data_path, sep='#')
    valid_pairs = valid_pairs.reset_index()
    valid_pairs.columns = ['left', 'right']

    # Extract the validation set from the train set
    valid_rows = []
    for ix, valid_pair in valid_pairs.iterrows():
        valid_row = train[(train['id_left'] == valid_pair[0]) & (train['id_right'] == valid_pair[1])]
        assert len(valid_row) == 1
        valid_rows.append(valid_row)

    valid = pd.concat(valid_rows)
    train = train[~train.index.isin(valid.index)]

    return train, valid


def clean_dataset(df):
    # Reformat train and validation sets
    feat_cols = ['id', 'title', 'description', 'brand', 'price']
    label = 'label'
    cols = [f'{c}_{suffix}' for suffix in ['left', 'right'] for c in feat_cols]
    cols += [label]
    feat_map = {f'{c}_left': f'left_{c}' for c in feat_cols}
    feat_map.update({f'{c}_right': f'right_{c}' for c in feat_cols})

    out_df = df[cols]
    out_df = out_df.rename(columns=feat_map)

    return out_df


if __name__ == '__main__':
    in_dir = r'C:\Users\matte\Downloads'
    out_dir = r'C:\Users\matte\PycharmProjects\bert-attention-for-em\data'
    use_case_type = 'xlarge'  # xlarge, large, medium, small
    use_case = 'Watches'  # Computers, Cameras, Shoes, Watches

    train_data_path = os.path.join(
        in_dir, use_case, f'{use_case.lower()}_train', f'{use_case.lower()}_train_{use_case_type}.json.gz'
    )
    valid_data_path = os.path.join(
        in_dir, use_case, f'{use_case.lower()}_valid', f'{use_case.lower()}_valid_{use_case_type}.csv'
    )
    test_data_path = os.path.join(in_dir, use_case, f'{use_case.lower()}_gs.json.gz')

    train, valid = split_train_valid(train_data_path, valid_data_path)
    test = pd.read_json(test_data_path, compression='gzip', lines=True)

    train = clean_dataset(train)
    valid = clean_dataset(valid)
    test = clean_dataset(test)

    print(f"TRAIN: {train.shape}")
    print(f"VALID: {valid.shape}")
    print(f"TEST: {test.shape}")

    train.to_csv(os.path.join(out_dir, use_case_type.capitalize(), use_case, 'train.csv'), index=False)
    valid.to_csv(os.path.join(out_dir, use_case_type.capitalize(), use_case, 'valid.csv'), index=False)
    test.to_csv(os.path.join(out_dir, use_case_type.capitalize(), use_case, 'test.csv'), index=False)

    # valid_integrated_data = convert_matching_pairs_to_integrated_dataset(
    #     data=valid,
    #     left_id='left_id',
    #     right_id='right_id',
    #     match_label='label',
    #     out_entity_col_id='entity_id'
    # )
    # valid_num_cliques = len(valid_integrated_data['entity_id'].unique()) if len(valid_integrated_data) > 0 else 0
    # print(valid_num_cliques)
    #
    # test_integrated_data = convert_matching_pairs_to_integrated_dataset(
    #     data=test,
    #     left_id='left_id',
    #     right_id='right_id',
    #     match_label='label',
    #     out_entity_col_id='entity_id'
    # )
    # test_num_cliques = len(test_integrated_data['entity_id'].unique()) if len(test_integrated_data) > 0 else 0
    # print(test_num_cliques)
