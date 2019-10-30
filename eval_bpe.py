import argparse
import unicodedata


def reverse_bpe(subwords, p_arc, p_rel):
    ids = []
    sub_ids = []
    for i, w in enumerate(subwords):
        if not w.endswith("@@"):
            sub_ids.append(i)
            ids.append(sub_ids)
            sub_ids = []
        else:
            sub_ids.append(i)
    spans = [i for i, x in enumerate(ids) for y in range(len(ids[i]))]
    words, arcs, rels = [None] * len(ids), [None] * len(ids), [None] * len(ids)
    for i in range(len(ids)):
        w = [subwords[x] for x in ids[i]]
        w = [x[:-2] if x.endswith("@@") else x for x in w]
        s = "".join(w)
        words[i] = s
        if p_arc[ids[i][-1]] == -1:
            arcs[i] = -1
            rels[i] = "none"
        elif p_arc[ids[i][-1]] == 0:
            arcs[i] = 0
            rels[i] = "root"
        else:
            arcs[i] = spans[p_arc[ids[i][-1]] - 1] + 1
            rels[i] = p_rel[ids[i][-1]]
    return words, arcs, rels
    

def evaluate(b_sen, g_sen):
    _, sub_words, _, _, _, _, p_arcs, p_rels, _, _ = zip(*b_sen)
    _, words, _, _, _, _, g_arcs, g_rels, _, _ = zip(*g_sen)
    p_arcs, g_arcs = list(map(int, p_arcs)), list(map(int, g_arcs))
    p_words, p_arcs, p_rels = reverse_bpe(sub_words, p_arcs, p_rels)
    assert list(p_words) == list(words)
    correct_arc = sum(x==y for x,y in zip(p_arcs, g_arcs) if y != -1)
    correct_rel = sum(x==y and a==b for x,y,a,b in zip(p_arcs, g_arcs, p_rels, g_rels) if y != -1)
    total = sum(x != -1 for x in g_arcs)
    return total, correct_arc, correct_rel

def is_punct(word):
    return all(unicodedata.category(char).startswith('P') for char in word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get raw sentences")
    parser.add_argument(
        "--bpe_file", default="../fairseq-master/pred.conllx", help="input file"
    )
    parser.add_argument(
        "--gold_file", default="../data/CODT/test.conll", help="output file"
    )

    args = parser.parse_args()

    bpe_file = open(args.bpe_file, "r", encoding="utf-8")
    gold_file = open(args.gold_file, "r", encoding="utf-8")
    b_sen, g_sen = [], []
    b_finish, g_finish = False, False
    total, correct_arcs, correct_rels = 0, 0, 0
    while True:
        if not b_finish:
            b_line = bpe_file.readline()
            if b_line == "\n":
                b_finish = True
            else:
                b_sen.append(b_line.strip().split())
        if not g_finish:
            g_line = gold_file.readline()
            if g_line == "\n":
                g_finish = True
            else:
                g_sen.append(g_line.strip().split())

        if b_finish and g_finish:
            t,a,r = evaluate(b_sen, g_sen)
            total += t
            correct_arcs += a
            correct_rels += r
            b_sen, g_sen = [], []
            b_finish, g_finish = False, False

        if not b_line:
            assert not g_line
            break

    uas = correct_arcs / total
    las = correct_rels / total
    print(total)
    print(f"uas:{uas:.2%}, las:{las:.2%}")
    bpe_file.close()
    gold_file.close()

