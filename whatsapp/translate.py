import logging


from whatsapp.libbots.parser import Parser


def _main():
    try:
        logging.basicConfig(
            format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
        )
        file = "whatsapp/data/raw/chat.txt"
        out_file = "whatsapp/data/processed/chat_english_2.txt"
        parser = Parser(file)
        parser.translate(out_file)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
