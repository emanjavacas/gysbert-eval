
import os
import json
import glob

source = "./data/ner/historic-domain-adaptation-icdar/processed/*.csv"
target = "./data/ner/historic-domain-adaptation-icdar/json/"

for file in glob.glob(source):
    sents, tokens, ners = [], [], []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sents.append({'token': tokens, 'ne_tags': ners, 'id': len(sents)})
                    tokesn, ners = [], []
            else:
                token, ner = line.split('\t')
                tokens.append(token)
                ners.append(ner)

    with open(target + os.path.basename(file).split('.')[0] + ".json", "w") as f:
        for sent in sents:
            f.write(json.dumps(sent) + "\n")

