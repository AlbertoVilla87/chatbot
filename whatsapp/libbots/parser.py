import os
import re
import numpy as np
import pandas as pd
import logging
import itertools
import collections
from tqdm import tqdm
import pickle
from nltk.tokenize import word_tokenize

from whatsapp.libbots.cleaner import Cleaner
from whatsapp.libbots.translation import Translator

UNKNOWN_TOKEN = "#UNK"
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"
MIN_TOKEN_FEQ = 20
SHUFFLE_SEED = 5871
EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"
MAX_USERS = 2
SEPARATOR = "$$$$$$"
MAX_TOKENS = 50

DIALS_PATH = "whatsapp/data/processed/dials.csv"

ASK = "ask"
REPLAY = "replay"
ASK_RAW = "ask_raw"
REPLAY_RAW = "replay_raw"

pd.options.mode.chained_assignment = None  # default='warn'


class Parser:
    def __init__(self, file: str):
        """_summary_
        :param file: _description_
        :type file: str
        """
        self.file = file
        self.dialogues_raw = self._read_txt()
        self.validate_content()

    def load_generative(self):
        """_summary_"""
        dialogs = self.dialogues_to_pairs_no_token()
        return dialogs

    def translate(self, file_path: str):
        """_summary_
        :param file_path: _description_
        :type file_path: str
        """
        with open(file_path, "w") as file:
            for string in tqdm(self.dialogues_raw[56691:]):
                trans = Translator.translate_sp_en_query(string)
                file.write(trans + "\n")

    def load_data_with_augmented(self, data_or: pd, data_au: pd):
        """_summary_
        :param self: _description_
        :type self: _type_
        :param pd: _description_
        :type pd: _type_
        :return: _description_
        :rtype: _type_
        """
        data_or = data_or.dropna()
        data_au = data_au.dropna()
        data_or[ASK] = data_or[ASK].apply(lambda x: eval(x))
        data_or[REPLAY] = data_or[REPLAY].apply(lambda x: eval(x))
        data_au[ASK] = data_au[ASK].apply(lambda x: Cleaner.clean(x))
        data_au[REPLAY] = data_au[REPLAY].apply(lambda x: Cleaner.clean(x))
        data_au[ASK] = data_au[ASK].apply(lambda x: word_tokenize(x))
        data_au[REPLAY] = data_au[REPLAY].apply(lambda x: word_tokenize(x))
        dials_au = self._convert_list_tuple(data_au)
        dials_or = self._convert_list_tuple(data_or)
        dials_pairs = dials_au + dials_or
        freq_set = self._freq_words(dials_pairs)
        dials_dict = self._dials_pairs_dict(dials_pairs, freq_set)
        return dials_pairs, dials_dict

    def load_data(self):
        """_summary_"""
        dials_pairs = self.dialogues_to_pairs()
        freq_set = self._freq_words(dials_pairs)
        dials_dict = self._dials_pairs_dict(dials_pairs, freq_set)
        return dials_pairs, dials_dict

    def encode_phrase_pairs(self, phrase_pairs, emb_dict, filter_unknows=True):
        """
        Convert list of phrase pairs to training data
        :param phrase_pairs: _description_
        :type phrase_pairs: _type_
        :param emb_dict: _description_
        :type emb_dict: _type_
        :param filter_unknows: _description_, defaults to True
        :type filter_unknows: bool, optional
        :return: _description_
        :rtype: _type_
        """
        unk_token = emb_dict[UNKNOWN_TOKEN]
        result = []
        for p1, p2 in phrase_pairs:
            p = Parser.encode_words(p1, emb_dict), Parser.encode_words(p2, emb_dict)
            if unk_token in p[0] or unk_token in p[1]:
                continue
            result.append(p)
        return result

    def dialogues_to_pairs(self):
        """_summary_"""
        users = self._get_users()
        user_one = self._get_dialog_user(users[0])
        user_two = self._get_dialog_user(users[1])
        num_dialogs = min(len(user_one), len(user_two))
        dials_pairs = []
        for index in range(0, num_dialogs):
            dials_pairs.append((user_one[index], user_two[index]))
        return dials_pairs

    def dialogues_to_pairs_no_token(self):
        """_summary_"""
        users = self._get_users()
        user_one = self._get_dialog_user_no_token(users[0])
        user_two = self._get_dialog_user_no_token(users[1])
        num_dialogs = min(len(user_one), len(user_two))
        dials_pairs = []
        for index in range(0, num_dialogs):
            if (len(user_one[index]) > 0) & (len(user_two[index]) > 0):
                dials_pairs.append((user_one[index], user_two[index]))
        return dials_pairs

    def validate_content(self):
        """_summary_"""
        names = self._get_users()
        print(names)
        if len(names) != MAX_USERS:
            raise Exception("Number of users is distint to 2")

    def split_train_test(self, data, train_ratio=0.95):
        count = int(len(data) * train_ratio)
        return data[:count], data[count:]

    @staticmethod
    def iterate_batches(data: list, batch_size: int):
        """_summary_
        :param data: _description_
        :type data: list
        :param batch_size: _description_
        :type batch_size: int
        :yield: _description_
        :rtype: _type_
        """
        assert isinstance(data, list)
        assert isinstance(batch_size, int)

        ofs = 0
        while True:
            batch = data[ofs * batch_size : (ofs + 1) * batch_size]
            if len(batch) <= 1:
                break
            yield batch
            ofs += 1

    @staticmethod
    def save_emb_dict(dir_name, emb_dict):
        with open(os.path.join(dir_name, EMB_DICT_NAME), "wb") as fd:
            pickle.dump(emb_dict, fd)

    @staticmethod
    def load_emb_dict(dir_name):
        with open(os.path.join(dir_name, EMB_DICT_NAME), "rb") as fd:
            return pickle.load(fd)

    @staticmethod
    def decode_words(indices, rev_emb_dict):
        return [rev_emb_dict.get(idx, UNKNOWN_TOKEN) for idx in indices]

    @staticmethod
    def encode_words(words, emb_dict):
        """
        Convert list of words into list of embeddings indices, adding our tokens
        :param words: list of strings
        :param emb_dict: embeddings dictionary
        :return: list of IDs
        """
        res = [emb_dict[BEGIN_TOKEN]]
        unk_idx = emb_dict[UNKNOWN_TOKEN]
        for w in words:
            idx = emb_dict.get(w.lower(), unk_idx)
            res.append(idx)
        res.append(emb_dict[END_TOKEN])
        return res

    @staticmethod
    def save_dials(dials: list):
        data = pd.DataFrame(dials, columns=[ASK, REPLAY])
        data[ASK_RAW] = data[ASK].apply(lambda x: " ".join(x))
        data[REPLAY_RAW] = data[REPLAY].apply(lambda x: " ".join(x))
        data.to_csv(DIALS_PATH, sep=";", index=False)

    @staticmethod
    def group_train_data(training_data):
        """
        Group training pairs by first phrase
        :param training_data: list of (seq1, seq2) pairs
        :return: list of (seq1, [seq*]) pairs
        """
        groups = collections.defaultdict(list)
        for p1, p2 in training_data:
            l = groups[tuple(p1)]
            l.append(p2)
        return list(groups.items())

    def _read_txt(self):
        with open(self.file) as f:
            lines = f.readlines()
        return lines

    def _get_name(self, comment: str):
        """_summary_
        :param comment: _description_
        :type comment: str
        """
        name_pattern = r"\[(?:[^,]+),[^]]+\]\s([^:]+):"
        match = re.search(name_pattern, comment)
        if match:
            name = match.group(1)
        else:
            name = ""
        return name

    def _get_dialog(self, comment: str):
        """_summary_
        :param comment: _description_
        :type comment: str
        """
        dialog_pattern = r":\s(.+)"
        match = re.search(dialog_pattern, comment)
        if match:
            dialog = match.group(1)
        else:
            dialog = comment
        dialog = Cleaner.clean(dialog)
        dialog = dialog.lower()
        return dialog

    def _get_dialog(self, comment: str):
        """_summary_
        :param comment: _description_
        :type comment: str
        """
        dialog_pattern = r":\s(.+)"
        match = re.search(dialog_pattern, comment)
        if match:
            dialog = match.group(1)
        else:
            dialog = comment
        dialog = Cleaner.clean(dialog)
        dialog = dialog.lower()
        return dialog

    def _get_dialog_generative(self, comment: str):
        """_summary_
        :param comment: _description_
        :type comment: str
        """
        dialog_pattern = r":\s(.+)"
        match = re.search(dialog_pattern, comment)
        if match:
            dialog = match.group(1)
        else:
            dialog = comment
        dialog = dialog.lower()
        dialog = Cleaner.clean_blacklist(dialog)
        return dialog

    def _get_users(self):
        """_summary_"""
        names = self._get_names()
        names = np.unique(names)
        return names

    def _get_names(self):
        """_summary_"""
        names = []
        for dial in self.dialogues_raw:
            name = self._get_name(dial)
            if name != "":
                names.append(name)
        return names

    def _get_dialog_user(self, user: str):
        """_summary_"""
        dials = []
        for dial in self.dialogues_raw:
            if user in dial:
                dials.append(SEPARATOR)
            else:
                dials.append(self._get_dialog(dial))
        dials = " ".join(dials)
        dials = dials.split(SEPARATOR)
        dials = [word_tokenize(dial.strip()) for dial in dials if dial != " "]
        return dials

    def _get_dialog_user_no_token(self, user: str):
        """_summary_
        :param user: _description_
        :type user: str
        """
        """_summary_"""
        dials = []
        for dial in self.dialogues_raw:
            if user in dial:
                dials.append(SEPARATOR)
            else:
                dials.append(self._get_dialog_generative(dial))
        dials = " ".join(dials)
        dials = dials.split(SEPARATOR)
        dials = [dial.strip() for dial in dials if dial != " "]
        return dials

    def _freq_words(self, dials_pairs: list, min_token_freq=MIN_TOKEN_FEQ):
        """_summary_
        :param dials_pairs: _description_
        :type dials_pairs: list
        :param min_token_freq: _description_, defaults to MIN_TOKEN_FEQ
        :type min_token_freq: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        word_counts = collections.Counter()
        for dial_one, dial_two in tqdm(dials_pairs, desc="frequency"):
            words = dial_one + dial_two
            word_counts.update(words)
        freq_set = set(
            map(
                lambda p: p[0],
                filter(
                    lambda p: p[1] >= min_token_freq,
                    word_counts.items(),
                ),
            )
        )
        logging.info(
            "Data has %d uniq words, %d of them occur more than %d",
            len(word_counts),
            len(freq_set),
            min_token_freq,
        )
        return freq_set

    def _dials_pairs_dict(self, dials_pairs, freq_set):
        """
        Return the dict of words in the dialogues mapped to their IDs
        :param dials_pairs: list of (phrase, phrase) pairs
        :return: dict
        """
        res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
        next_id = 3
        for p1, p2 in tqdm(dials_pairs, desc="dictionary"):
            for w in map(str.lower, itertools.chain(p1, p2)):
                if w not in res and w in freq_set:
                    res[w] = next_id
                    next_id += 1
        return res

    def _convert_list_tuple(self, data):
        """_summary_
        :param data: _description_
        :type data: _type_
        """
        dialogs = []
        for _, row in data.iterrows():
            dialogs.append((row[ASK], row[REPLAY]))
        return dialogs
