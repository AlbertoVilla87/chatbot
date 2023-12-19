#!/usr/bin/env python3
import os
import argparse
import logging
import torch

from whatsapp.libbots import parser
from whatsapp.libbots.parser import Parser
from whatsapp.libbots import model, utils


def words_to_words(words, emb_dict, rev_emb_dict, net, use_sampling=False):
    tokens = Parser.encode_words(words, emb_dict)
    input_seq = model.pack_input(tokens, net.emb)
    enc = net.encode(input_seq)
    end_token = emb_dict[parser.END_TOKEN]
    if use_sampling:
        _, out_tokens = net.decode_chain_sampling(
            enc, input_seq.data[0:1], seq_len=parser.MAX_TOKENS, stop_at_token=end_token
        )
    else:
        _, out_tokens = net.decode_chain_argmax(
            enc, input_seq.data[0:1], seq_len=parser.MAX_TOKENS, stop_at_token=end_token
        )
    if out_tokens[-1] == end_token:
        out_tokens = out_tokens[:-1]
    out_words = Parser.decode_words(out_tokens, rev_emb_dict)
    return out_words


def process_string(s, emb_dict, rev_emb_dict, net, use_sampling=False):
    out_words = words_to_words(
        words, emb_dict, rev_emb_dict, net, use_sampling=use_sampling
    )
    print(" ".join(out_words))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
    )

    STRING = "con hermosas venas"
    model_name = "saves/MyGirlBotCE/epoch_100_0.868_0.016.dat"

    emb_dict = Parser.load_emb_dict(os.path.dirname(model_name))

    net = model.PhraseModel(
        emb_size=model.EMBEDDING_DIM,
        dict_size=len(emb_dict),
        hid_size=model.HIDDEN_STATE_SIZE,
    )
    net.load_state_dict(torch.load(model_name))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    words = utils.tokenize(STRING)
    words = words_to_words(words, emb_dict, rev_emb_dict, net, use_sampling=True)
    print(utils.untokenize(words))
