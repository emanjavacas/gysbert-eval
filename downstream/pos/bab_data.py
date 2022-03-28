
from email.policy import default
import json
import re
import pandas as pd
import os
import glob
from lxml import etree
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vrts', default="/home/manjavacasema/data/downstream/gysbert-eval/pos/data/BaB2.0/BaBVerticalTextv2.0/*xml")
    parser.add_argument('--xmls', default="/home/manjavacasema/data/downstream/gysbert-eval/pos/data/BaB2.0/BaBXMLv2.0/*xml")
    args = parser.parse_args()

    vrts = glob.glob(args.vrts)
    xmls = glob.glob(args.xmls)

    # vrts = [os.path.basename(f) for f in vrts]
    # xmls = [os.path.basename(f) for f in xmls]

    # set(vrts).difference(set(xmls))
    # set(xmls).difference(set(vrts))

    dates_path = './data/pos/BaB2.0/BaBXMLv2.0-dates.csv'
    if not os.path.isfile(dates_path):
        dates = []
        for f in xmls:
            tree = etree.parse(f)
            assert len(tree.xpath('//interpGrp[@type="witnessYear_from"]')) ==1
            date = tree.xpath('//interpGrp[@type="witnessYear_from"]')[0]
            dates.append({'file_id': os.path.basename(f), 'date': int(date.find('interp').attrib['value'])})
        pd.DataFrame.from_dict(dates).to_csv(dates_path, index=None)
    else:
        dates = pd.read_csv(dates_path)
    import sys

    # splits
    train, test = train_test_split(dates.index, test_size=0.10, random_state=1001)
    train, dev = train_test_split(train, test_size=0.05, random_state=1001)


    def get_lines(f, max_length=100):
        # remove any in-line markup like: "loe<uncertain>s</uncertain>"
        with open(f) as inp:
            output = []
            for orig_line in inp:
                line = re.sub(r"<[^>]+>", "", orig_line.strip())
                if not line:
                    continue
                if len(output) >= max_length:
                    tokens, pos, lemmas = zip(*output)
                    yield {'tokens': tokens, 'pos_tags': pos, "lemma": lemmas}
                    output = []
                try:
                    tok, pos, lemma = line.split('\t')
                    output.append((tok, pos, lemma))
                except:
                    print(orig_line)
                    continue
            if output:
                tokens, pos, lemmas = zip(*output)
                yield {'tokens': tokens, 'pos_tags': pos, "lemma": lemmas}

    output_dir = "./data/pos/BaB2.0/splits/"
    root = os.path.dirname(args.vrts)
    for split, idxs in zip(['train', 'test', 'dev'], [train, test, dev]):
        if split == 'train':
            for n_docs in [10, 50, 100, 500]:
                target = os.path.join(output_dir, "BaB2.0-{}".format(n_docs))
                os.makedirs(target, exist_ok=True)
                with open(os.path.join(target, 'train.json'), "w+") as outf:
                    for f in dates.iloc[idxs]['file_id'].sample(n_docs):
                        full_path = os.path.join(root, f)
                        for line_num, line in enumerate(get_lines(full_path)):
                            line['id'] = f + "," + str(line_num + 1)
                            line['file'] = f
                            outf.write(json.dumps(line) + "\n")
        target = os.path.join(output_dir, "BaB2.0-full")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, '{}.json'.format(split)), "w+") as outf:
            for f in dates.iloc[idxs]['file_id']:
                full_path = os.path.join(root, f)
                for line_num, line in enumerate(get_lines(full_path)):
                    line['id'] = f + "," + str(line_num + 1)
                    line['file'] = f
                    outf.write(json.dumps(line) + "\n")
