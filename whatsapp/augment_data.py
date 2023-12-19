import os
import logging
import pandas as pd

from whatsapp.libbots.augmenter import Augmenter

os.environ["CURL_CA_BUNDLE"] = ""


def _main():
    try:
        logging.basicConfig(
            format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
        )
        data_in = pd.read_csv("whatsapp/data/processed/dials.csv", sep=";")
        data_in = data_in.dropna()
        data_out = Augmenter.back_translate_data(data_in)
        data_out.to_csv("whatsapp/data/augmented/dials.csv", sep=";", index=False)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
