import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get raw sentences")
    parser.add_argument(
        "--in_file", default="/data/wjiang/data/NMT/ldc-data/test-bpe.en", help="input file"
    )
    parser.add_argument(
        "--out_file", default="/data/wjiang/data/NMT/ch-en-conllx/test-bpe-conllx.en", help="output file"
    )

    args = parser.parse_args()

    f_in = open(args.in_file, "r", encoding="utf-8")
    f_out = open(args.out_file, "w", encoding="utf-8")

    for line in f_in:
        words = line.strip().split()
        for i, word in enumerate(words):
            x = [
                str(i + 1),
                word,
                "_",
                "NN",
                "NN",
                "_",
                str(-1),
                "nmod",
                "_",
                "_",
            ]
            f_out.write("\t".join(x))
            f_out.write("\n")
        f_out.write("\n")

    f_in.close()
    f_out.close()
