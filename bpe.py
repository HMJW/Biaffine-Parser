import argparse

def mysplit(sen):
    result = []
    for x in sen.strip().split(" "):
        result += x.split("\xa0")
    return result

def bpe(raw, deps):
    words = [w[1] for w in deps]
    ids = []
    sub_ids = []
    for i, w in enumerate(raw):
        if not w.endswith("@@"):
            sub_ids.append(i)
            ids.append(sub_ids)
            sub_ids = []
        else:
            sub_ids.append(i)
    # assert len(ids) == len(deps)

    pos1 = [None] * len(raw)
    pos2 = [None] * len(raw)
    heads = [None] * len(raw)
    rels = [None] * len(raw)
    probs = [None] * len(raw)
    for id, dep in zip(ids, deps):
        if len(id) > 1:
            for i in id[:-1]:
                pos1[i] = dep[3]
                pos2[i] = dep[4]
                heads[i] = i + 1 + 1
                rels[i] = "subword"
                probs[i] = "1.0"
        pos1[id[-1]] = dep[3]
        pos2[id[-1]] = dep[4]
        if dep[6] == "0":
            heads[id[-1]] = 0
        elif dep[7] == "none":
            heads[id[-1]] = -1
        else:
            heads[id[-1]] = ids[int(dep[6]) - 1][-1] + 1
        rels[id[-1]] = dep[7]
        probs[id[-1]] = dep[-1]

    results = []
    for i in range(len(raw)):
        results.append(
            [
                str(i + 1),
                raw[i],
                "_",
                pos1[i],
                pos2[i],
                "_",
                str(heads[i]),
                rels[i],
                "_",
                str(probs[i])
            ]
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get raw sentences")
    parser.add_argument(
        "--in_file", default="../data/summary-corpus/train.bpe.src", help="input file"
    )
    parser.add_argument(
        "--train_file", default="../data/summary-corpus/train.ptb-bert.conllx.src", help="input file"
    )
    parser.add_argument(
        "--out_file", default="../data/summary-corpus/train.bpe.conllx.src", help="output file"
    )
    parser.add_argument(
        "--id", "-i", default=0, help="input file", type=int
    )

    args = parser.parse_args()

    f_in = open(args.in_file, "r", encoding="utf-8")
    train_in = open(args.train_file, "r", encoding="utf-8")
    f_out = open(args.out_file, "w", encoding="utf-8")

    bep_sens = []
    count = 0
    for line in f_in:
        bep_sens.append(mysplit(line))
        if count == args.id and args.id >= 0:
            break
        count += 1

    sentences, sentence = [], []
    count = 0
    for line in train_in:
        if line == "\n":
            sentences.append(sentence)
            sentence = []
            if count == args.id and args.id >= 0:
                break
            count += 1
        else:
            line = line.split()
            sentence.append(line)

    new_sen = []
    for i, (x, y) in enumerate(zip(bep_sens, sentences)):
        try:
            new_sen.append(bpe(x, y))
        except Exception as e:
            print(i)
    # new_sen = [bep(x, y) for x, y in zip(bep_sens, sentences)]
    for x in new_sen:
        for y in x:
            if y[1] == "" or len(y[1].split()) == 0:
                y[1] = "-"
            f_out.write("\t".join(y))
            f_out.write("\n")
        f_out.write("\n")

    f_in.close()
    train_in.close()
    f_out.close()
