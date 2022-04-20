

import pandas as pd
from sklearn.model_selection import train_test_split


def encode_data(tokenizer, sents, starts, ends, sym='[TGT]'):
    output_sents, spans = [], []
    for sent, char_start, char_end in zip(sents, starts, ends):
        # insert target symbols
        if sym is not None:
            sent = sent[:char_start] + '{} '.format(sym) + \
                sent[char_start:char_end] + ' {}'.format(sym) + sent[char_end:]
        output_sents.append(sent)

        sent = tokenizer.encode_plus(sent, return_offsets_mapping=True)
        # transform character indices to subtoken indices
        target_start = target_end = None
        if sym is not None:
            char_start += len(sym) + 1
            char_end += len(sym) + 1
        for idx, (token_start, token_end) in enumerate(sent['offset_mapping']):
            if token_start == char_start:
                target_start = idx
            if token_end == char_end:
                target_end = idx
        if target_start is None or target_end is None:
            raise ValueError
        spans.append((target_start, target_end + 1))

    # encode sentences
    encoded = tokenizer(output_sents, return_tensors='pt', padding=True)

    return encoded, spans


def generate_splits(df_source, targets, key):
    splits = []
    for lemma, subset in df_source[df_source['lemma'].isin(targets)].groupby('lemma'):
        # drop lemmas not in target frequency
        if lemma not in targets:
            continue
        # drop senses where we can't stratify
        senses = subset[key].value_counts()
        subset = subset[subset[key].isin(senses[senses >= 2].index)]
        if len(subset) < 2:
            continue
        # drop lemmas with only one sense
        senses = subset[key].value_counts()
        if len(senses) == 1:
            continue
        train, test = train_test_split(subset.index, stratify=subset[key], test_size=0.5)
        assert set(df_source.iloc[test][key]).difference(
            set(df_source.iloc[train][key])) == set()
        splits.append((train, test))

    return splits


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quotations', default='/Users/manjavacas/code/INT/WNT.quotations.csv')
    parser.add_argument('--entries', default='/Users/manjavacas/code/INT/WNT.entries.csv')
    args = parser.parse_args()

    df_source = pd.read_csv(args.quotations)
    entries = pd.read_csv(args.entries)

    # df_source = pd.read_csv('/Users/manjavacas/code/INT/WNT.quotations.csv', keep_default_na=False)
    # entries = pd.read_csv('/Users/manjavacas/code/INT/WNT.entries.csv')

    # add meta info
    df_source = pd.merge(
        left=df_source, right=entries[['id', 'lemma', 'result_type']], 
        left_on='entry_id', right_on='id')

    # merge entries with same lemma but different entry id
    for _, g in df_source.groupby('lemma'):
        if len(g['entry_id'].unique()) > 1:
            df_source.loc[g.index, 'sense_id'] = g.apply(lambda row: row['entry_id'] + '/' + row['sense_id'], axis=1)

    depths = set(df_source['sense_id'].apply(lambda row: len(row.split('/'))))
    for depth in depths:
        df_source['depth-{}'.format(depth)] = df_source['sense_id'].apply(
            lambda row: '/'.join(row.split('/')[:depth]))

    # min count
    counts = df_source['lemma'].value_counts()
    targets = counts[counts >= 50].index

    # zero_targets
    targets, zero_targets = train_test_split(targets, test_size=0.1, random_state=1001)

    *path, ext = args.quotations.split('.')

    for key in depths:
        key = 'depth-{}'.format(key)
        # training splits
        splits = generate_splits(df_source, targets, key)
        pd.concat([df_source.iloc[t] for t, _ in splits]).to_csv(
            '.'.join(path) + '-' + key + '-train.csv', index=None)
        pd.concat([df_source.iloc[t] for _, t in splits]).to_csv(
            '.'.join(path) + '-' + key + '-test.csv', index=None)
        # zero-shot splits
        splits = generate_splits(df_source, zero_targets, key)
        pd.concat([df_source.iloc[t] for t, _ in splits]).to_csv(
            '.'.join(path) + '-' + key + '-train.zero.csv', index=None)
        pd.concat([df_source.iloc[t] for _, t in splits]).to_csv(
            '.'.join(path) + '-' + key + '-test.zero.csv', index=None)


# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
# mpath = "/Users/manjavacas/Leiden/diachronic-token-embeddings/gysbert/GysBERT-1.5m"
# tokenizer = AutoTokenizer.from_pretrained(mpath)
# tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
# sents, starts, ends = zip(*df_source[['quote', 'start', 'end']].values)
# df_source['target'].apply(lambda row: row[-1]).value_counts()
# sym = '[TGT]'
# output_sents, spans, wrong = [], [], []
# for sent_idx, (sent, char_start, char_end) in enumerate(zip(sents, starts, ends)):
#     # insert target symbols
#     if sym is not None:
#         sent = sent[:char_start] + '{} '.format(sym) + \
#             sent[char_start:char_end] + ' {}'.format(sym) + sent[char_end:]
#     # output_sents.append(sent)

#     sent = tokenizer.encode_plus(sent, return_offsets_mapping=True)

#     # transform character indices to subtoken indices
#     target_start = target_end = None
#     if sym is not None:
#         char_start += len(sym) + 1
#         char_end += len(sym) + 1
#     for idx, (token_start, token_end) in enumerate(sent['offset_mapping']):
#         if token_start == char_start:
#             target_start = idx
#         if token_end == char_end:
#             target_end = idx
#     if target_start is None or target_end is None:
#         wrong.append(sent_idx)
#         # raise ValueError
#         spans.append(None)
#     else:
#         spans.append((target_start, target_end + 1))

# len(wrong)
# df_source.iloc[wrong]['target']
# # idx = wrong[10]
# # sent, char_start, char_end = sents[idx], starts[idx], ends[idx]
# # sent, char_start, char_end
# # target_start, target_end
# # wrong


# for i in range(1, 8):
#     print(pd.isna(df_source['depth-{}'.format(i)]).sum())
