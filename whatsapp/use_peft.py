import logging


from whatsapp.libbots.parser import Parser
from whatsapp.libbots.generative import GenerativeWhatsAppAnswer

DEVICE = "cpu"
MODEL_BASE = "whatsapp/flan-t5-small"
MODEL_PEFT = "peft-albertobot-checkpoint-local"
MODEL_ORIGINAL = "google/flan-t5-base"


def _main():
    try:
        logging.basicConfig(
            format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
        )
        start_prompt = "Answer this question.\n\n"
        end_prompt = ""
        dial = "What do you say?"
        prompt = start_prompt + dial
        print(prompt)
        generative = GenerativeWhatsAppAnswer(MODEL_ORIGINAL, MODEL_BASE, MODEL_PEFT)
        out = generative.answer(prompt)
        print(out)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
