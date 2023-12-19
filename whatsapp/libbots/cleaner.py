# TODO: Simplify remove emojies https://dencode.com/
# TODO: Use an LLM model to misspeling
import re
import unicodedata
from nltk.tokenize import word_tokenize

BLACK_LIST_WP = [
    "audio omitido",
    "\u200eimagen",
    "\u200esticker",
    "\u200evideo",
    "omitido",
    "omitida",
]


class Cleaner:
    @staticmethod
    def clean(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        :return: _description_
        :rtype: _type_
        """
        text = Cleaner.clean_laugh(text)
        text = Cleaner.clean_emojies(text)
        text = Cleaner.clean_duplicates(text)
        text = Cleaner.clean_blacklist(text)
        text = Cleaner.clean_accents(text)
        text = Cleaner.clean_no_info(text)
        text = Cleaner.clean_single_char(text)
        return text

    @staticmethod
    def clean_laugh(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        pattern = r"\b[ja]+\b"
        text = re.sub(pattern, "lol", text)
        return text

    @staticmethod
    def clean_emojies(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        pattern = r"😚+"
        text = re.sub(pattern, " 😚 ", text)
        pattern = r"😬+"
        text = re.sub(pattern, " 😬 ", text)
        pattern = r"🥰+"
        text = re.sub(pattern, " 😚 ", text)
        pattern = r"🚩+"
        text = re.sub(pattern, " 🚩 ", text)
        pattern = r"🤣+"
        text = re.sub(pattern, " 🤣 ", text)
        pattern = r"😅+"
        text = re.sub(pattern, " 😅 ", text)
        pattern = r"😳+"
        text = re.sub(pattern, " 😳 ", text)
        pattern = r"🙃+"
        text = re.sub(pattern, " 🙃 ", text)
        pattern = r"😆+"
        text = re.sub(pattern, " 😆 ", text)
        pattern = r"😝+"
        text = re.sub(pattern, " 😝 ", text)
        pattern = r"😘+"
        text = re.sub(pattern, " 😘 ", text)
        pattern = r"😁+"
        text = re.sub(pattern, " 😁 ", text)
        pattern = r"💪🏽+"
        text = re.sub(pattern, " 💪🏽 ", text)
        pattern = r"😍+"
        text = re.sub(pattern, " 😍 ", text)
        pattern = r"😐+"
        text = re.sub(pattern, " 😐 ", text)
        pattern = r"🦄+"
        text = re.sub(pattern, " 🦄 ", text)
        pattern = r"😒+"
        text = re.sub(pattern, " 😒 ", text)
        return text

    @staticmethod
    def clean_duplicates(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        pattern = r"!+"
        text = re.sub(pattern, " ! ", text)
        pattern = r"\?+"
        text = re.sub(pattern, " ? ", text)
        pattern = r"\.+"
        text = re.sub(pattern, " . ", text)
        return text

    @staticmethod
    def clean_blacklist(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        for blackl in BLACK_LIST_WP:
            text = text.replace(blackl, "")
        return text

    @staticmethod
    def clean_underscore(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        text = text.replace("▁", " ").strip()
        return text

    @staticmethod
    def clean_no_info(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        words = word_tokenize(text)
        words = [re.sub(r"[^\wñ]", "", word) for word in words]
        text = " ".join(words)
        return text

    @staticmethod
    def clean_single_char(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        words = word_tokenize(text)
        words = [word for word in words if len(word) > 1]
        text = " ".join(words)
        return text

    @staticmethod
    def clean_accents(text: str):
        """_summary_
        :param text: _description_
        :type text: str
        """
        text = text.replace("á", "a")
        text = text.replace("é", "e")
        text = text.replace("í", "i")
        text = text.replace("ó", "o")
        text = text.replace("ú", "u")
        return text
