import argparse


def bep(raw, deps):
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
    assert len(ids) == len(deps)

    pos1 = [None] * len(raw)
    pos2 = [None] * len(raw)
    heads = [None] * len(raw)
    rels = [None] * len(raw)
    for id, dep in zip(ids, deps):
        if len(id) > 1:
            for i in id[:-1]:
                pos1[i] = dep[3]
                pos2[i] = dep[4]
                heads[i] = i + 1 + 1
                rels[i] = "app"
        pos1[id[-1]] = dep[3]
        pos2[id[-1]] = dep[4]
        if dep[6] == "0":
            heads[id[-1]] = 0
        elif dep[7] == "none":
            heads[id[-1]] = -1
        else:
            heads[id[-1]] = ids[int(dep[6]) - 1][-1] + 1
        rels[id[-1]] = dep[7]

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
                "_",
            ]
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get raw sentences")
    parser.add_argument(
        "--in_file", default="/data/wjiang/data/CODT/test.bpe.txt", help="input file"
    )
    parser.add_argument(
        "--train_file", default="/data/wjiang/data/CODT/test.conll", help="input file"
    )
    parser.add_argument(
        "--out_file", default="/data/wjiang/data/CODT/test-bpe.conll", help="output file"
    )

    args = parser.parse_args()

    f_in = open(args.in_file, "r", encoding="utf-8")
    train_in = open(args.train_file, "r", encoding="utf-8")
    f_out = open(args.out_file, "w", encoding="utf-8")

    bep_sens = [line.split() for line in f_in]

    sentences, sentence = [], []
    for line in train_in:
        if line == "\n":
            sentences.append(sentence)
            sentence = []
        else:
            line = line.split()
            sentence.append(line)

    new_sen = [bep(x, y) for x, y in zip(bep_sens, sentences)]
    for x in new_sen:
        for y in x:
            f_out.write("\t".join(y))
            f_out.write("\n")
        f_out.write("\n")

    f_in.close()
    train_in.close()
    f_out.close()
