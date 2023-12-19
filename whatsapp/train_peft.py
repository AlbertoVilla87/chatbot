import logging


from whatsapp.libbots.parser import Parser
from whatsapp.libbots.generative import GenerativeWhatsAppTrain

DEVICE = "cpu"
MODEL = "whatsapp/flan-t5-small"


def _main():
    try:
        logging.basicConfig(
            format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
        )
        file = "whatsapp/data/processed/chat_english.txt"
        parser = Parser(file)
        dialogs = parser.dialogues_to_pairs_no_token()
        generative = GenerativeWhatsAppTrain(MODEL)
        generative.prepare_data(dialogs)
        # generative.train()

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
