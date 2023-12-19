# https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb

import pandas as pd
import nlpaug.augmenter.word as naw
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

from whatsapp.libbots.cleaner import Cleaner
from whatsapp.libbots import parser

BERT = "whatsapp/bert-base-uncased"
DBERT = "whatsapp/distilbert-base-uncased"
ROB = "whatsapp/roberta-base"
OPUS_EN_ES = "whatsapp/opus-mt-en-es"
OPUS_ES_EN = "whatsapp/opus-mt-es-en"


class Augmenter:
    @staticmethod
    def substitute_word_bert(text: str, aug_p: float = 0.3) -> str:
        """
        Substitute word by contextual word embeddings using BERT
        :param text: _description_
        :type text: str
        :param aug_p: _description_, defaults to 0.3
        :type aug_p: float, optional
        :return: _description_
        :rtype: str
        """
        text = Augmenter._action(text, BERT, aug_p, action="substitute")
        return text[0].lower()

    @staticmethod
    def substitute_word_dbert(text: str, aug_p: float = 0.3) -> str:
        """
        Substitute word by contextual word embeddings using DISTIL-BERT
        :param text: _description_
        :type text: str
        :param aug_p: _description_, defaults to 0.3
        :type aug_p: float, optional
        :return: _description_
        :rtype: str
        """
        text = Augmenter._action(text, DBERT, aug_p, action="substitute")
        return text[0].lower()

    @staticmethod
    def substitute_word_rob(text: str, aug_p: float = 0.3) -> str:
        """
        Substitute word by contextual word embeddings using ROBERTA
        :param text: _description_
        :type text: str
        :param aug_p: _description_, defaults to 0.3
        :type aug_p: float, optional
        :return: _description_
        :rtype: str
        """
        text = Augmenter._action(text, ROB, aug_p, action="substitute")
        return text[0].lower()

    @staticmethod
    def insert_word_bert(text: str, aug_p: float = 0.3) -> str:
        """
        Substitute word by contextual word embeddings using BERT
        :param text: _description_
        :type text: str
        :param aug_p: _description_, defaults to 0.3
        :type aug_p: float, optional
        :return: _description_
        :rtype: str
        """
        text = Augmenter._action(text, BERT, aug_p, action="insert")
        return text[0].lower()

    @staticmethod
    def insert_word_dbert(text: str, aug_p: float = 0.3) -> str:
        """
        Substitute word by contextual word embeddings using DISTIL-BERT
        :param text: _description_
        :type text: str
        :param aug_p: _description_, defaults to 0.3
        :type aug_p: float, optional
        :return: _description_
        :rtype: str
        """
        text = Augmenter._action(text, DBERT, aug_p, action="insert")
        return text[0].lower()

    @staticmethod
    def insert_word_rob(text: str, aug_p: float = 0.3) -> str:
        """
        Substitute word by contextual word embeddings using ROBERTA
        :param text: _description_
        :type text: str
        :param aug_p: _description_, defaults to 0.3
        :type aug_p: float, optional
        :return: _description_
        :rtype: str
        """
        text = Augmenter._action(text, ROB, aug_p, action="insert")
        return text[0].lower()

    @staticmethod
    def back_translate_data(data: pd) -> pd:
        """_summary_
        :param data: _description_
        :type data: pd
        :return: _description_
        :rtype: pd
        """
        source_tokenizer = MarianTokenizer.from_pretrained(OPUS_ES_EN)
        target_tokenizer = MarianTokenizer.from_pretrained(OPUS_EN_ES)
        source_model = MarianMTModel.from_pretrained(OPUS_ES_EN)
        target_model = MarianMTModel.from_pretrained(OPUS_EN_ES)
        tqdm.pandas()
        ask = data[parser.ASK_RAW].progress_apply(
            lambda x: Augmenter.back_translate(
                x,
                source_tokenizer,
                target_tokenizer,
                source_model,
                target_model,
            )
        )
        tqdm.pandas()
        replay = data[parser.REPLAY_RAW].progress_apply(
            lambda x: Augmenter.back_translate(
                x,
                source_tokenizer,
                target_tokenizer,
                source_model,
                target_model,
            )
        )
        data_out = pd.DataFrame.from_dict(
            {
                parser.ASK_RAW: data[parser.ASK_RAW].values,
                parser.ASK: ask,
                parser.REPLAY_RAW: data[parser.REPLAY_RAW],
                parser.REPLAY: replay,
            }
        )
        return data_out

    @staticmethod
    def back_translate(
        text: str,
        source_tokenizer: MarianTokenizer,
        target_tokenizer: MarianTokenizer,
        source_model: MarianMTModel,
        target_model: MarianMTModel,
        max_length: int = 100,
    ):
        """_summary_
        :param text: _description_
        :type text: str
        :param source_tokenizer: _description_
        :type source_tokenizer: MarianTokenizer
        :param target_tokenizer: _description_
        :type target_tokenizer: MarianTokenizer
        :param source_model: _description_
        :type source_model: MarianMTModel
        :param target_model: _description_
        :type target_model: MarianMTModel
        :param max_length: _description_, defaults to 100
        :type max_length: int, optional
        :return: _description_
        :rtype: _type_
        """
        # Translate the original sentence to the target language
        inputs = source_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = source_model.generate(**inputs, max_length=max_length)
        target_translation = target_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        # Translate the target translation back to the source language
        inputs = target_tokenizer(
            target_translation, return_tensors="pt", truncation=True
        )
        outputs = target_model.generate(**inputs, max_length=max_length)
        back_translated_sentence = source_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        back_translated_sentence = Cleaner.clean_underscore(back_translated_sentence)
        return back_translated_sentence.lower()

    def _action(text, model, aug_p, action):
        """_summary_
        :param model: _description_
        :type model: _type_
        :param text: _description_
        :param aug_p: _description_
        :type aug_p: int
        :type text: _type_
        """
        aug = naw.ContextualWordEmbsAug(
            model_path=model,
            action=action,
            aug_p=aug_p,
            top_k=1,
        )
        augmented_text = aug.augment(text)
        return augmented_text
