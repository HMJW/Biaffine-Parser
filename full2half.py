# -*- coding: utf-8 -*-

import argparse
import unicodedata

# FF00-FF5F -> 0020-007E
MAP = {'　': ' ', '！': '!', '＂': '"', '＃': '#', '＄': '$', '％': '%', '＆': '&', 
       '＇': "'", '（': '(', '）': ')', '＊': '*', '＋': '+', '，': ',', '－': '-', 
       '．': '.', '／': '/', 
       '０': '0', '１': '1', '２': '2', '３': '3', '４': '4', '５': '5', '６': '6', 
       '７': '7', '８': '8', '９': '9', 
       '：': ':', '；': ';', '＜': '<', '＝': '=', '＞': '>', '？': '?', '＠': '@',
       'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E', 'Ｆ': 'F', 'Ｇ': 'G',
       'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J', 'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N',
       'Ｏ': 'O', 'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T', 'Ｕ': 'U',
       'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z', 
       '［': '[', '＼': '\\', 
       '］': ']', '＾': '^', '＿': '_', '｀': '`',
       'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g',
       'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n',
       'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't', 'ｕ': 'u',
       'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z', 
       '｛': '{', '｜': '|', '｝': '}'}


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A']
               for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char)
               for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char)
               for char in token)


def tohalfwidth(token):
    return unicodedata.normalize('NFKC', token)


def full2half(fin, fout, narrow=False):
    r'''Convert full-width characters to half-width ones.

    Parameters:
        fin (str): the file to convert.
        fout (str): the file to save.
        narrow (bool):
            True if only convert the characters in the range [FF00, FF5F);
            False else.
    '''

    with open(fin, 'r') as f:
        lines = [l.strip() for l in f]
    if narrow:
        lines = [''.join(MAP.get(c, c) for c in l) for l in lines]
    else:
        lines = [tohalfwidth(l) for l in lines]
    with open(fout, 'w') as f:
        for l in lines:
            f.write(l + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert full-width characters to half-width ones.'
    )
    parser.add_argument('--fin', help='the file to convert', default="../data/CODT/test.conll")
    parser.add_argument('--fout', help='the file to save', default="../data/CODT/test.half.conll")
    parser.add_argument('--narrow', action='store_true',
                        help='only convert the characters in the above table')
    args = parser.parse_args()
    full2half(args.fin, args.fout, args.narrow)