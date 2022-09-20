from cgitb import text
import sys
import math
from typing import Type; sys.path.insert(0, "..")
import argparse
# from googletrans import Translator
from BackTranslation import BackTranslation
from text_utils import get_preprocessor
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        """Backtranslation""")
    parser.add_argument("--data_dir", type=str, default="/hdd2/lannliat/CP-VAE/data/gyafc/processed/")
    parser.add_argument("--tmp", type=str, default="fr")
    args = parser.parse_args()
    return args

# def backtranslate_text(txt, translator, src="en", dest="fr"):
#     print(len(txt))
#     try: 
#         txt_tr = translator.translate(txt, dest=dest)
#         txt_bt = translator.translate(txt_tr, dest=src)
#     except TypeError: # too long
#         mp = len(txt) // 2
#         txt_1 = txt[:mp]
#         txt_2 = txt[mp:]
#         txt_1_bt = backtranslate_text(txt_1, translator)
#         txt_2_bt = backtranslate_text(txt_2, translator)

#         txt_bt = txt_1_bt + txt_2_bt
#     return txt_bt

def backtranslate_text(txt, translator, tmp="fr"):
    try:
        txt_bt = translator.translate(txt, src='en', tmp=tmp).result_text
    except TypeError:  # too long
        print(f"Translated text too long. Translating by halves...")
        mp = len(txt) // 2  # it cuts text abrubtly, but should be okay since our docs are long; 1-2 words messing up is fine
        txt_1 = txt[:mp]
        txt_2 = txt[mp:]

        txt_1_bt = backtranslate_text(txt_1, translator)
        txt_2_bt = backtranslate_text(txt_2, translator)

        txt_bt = f"{txt_1_bt}{txt_2_bt}"
    return txt_bt

def main(args):
    preprocessor = get_preprocessor(args.data_dir)(args.data_dir)

    trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})

    datasets = preprocessor.load_datasets()
    datasets_bt = []
    for dataset in datasets:
        print(f"Backtranslating...")
        sents, labels = dataset
        sents_bt = [backtranslate_text(sent, trans) for sent in tqdm(sents)]
        # sents_bt = backtranslate_text(sents, trans, args.src, args.dest)
        datasets_bt.append((sents_bt, labels))
    
    preprocessor.write_datasets(datasets_bt, append_name="bt")    

if __name__ == "__main__":
    opt = get_args()
    main(opt)
