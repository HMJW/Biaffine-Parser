import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get raw sentences")
    parser.add_argument("--in_file", default="/data/wjiang/data/CODT/test.conll", help="input file")
    parser.add_argument("--out_file", default="/data/wjiang/data/CODT/test.raw.txt", help="output file")

    args = parser.parse_args()

    f_in = open(args.in_file, "r", encoding="utf-8")
    f_out = open(args.out_file, "w", encoding="utf-8")
    sentences = []
    for line in f_in:
        if line == "\n":
            f_out.write(" ".join(sentences))
            f_out.write("\n")
            sentences = []
        else:
            sentences.append(line.split()[1])
    f_in.close()
    f_out.close()