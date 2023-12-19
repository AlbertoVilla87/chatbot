import os
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F

from whatsapp.libbots import parser
from whatsapp.libbots.parser import Parser
from whatsapp.libbots import model
from whatsapp.libbots import optimizer_rl
from whatsapp.libbots.optimizer_rl import OptimizerRL

DEVICE = "cpu"


def _main():
    try:
        logging.basicConfig(
            format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
        )
        name = "MyGirlBotRL"
        file = "whatsapp/data/raw/chat.txt"
        base_model = "saves/MyGirlBotCE/epoch_100_0.922_0.010.dat"

        saves_path = os.path.join(optimizer_rl.SAVES_DIR, name)
        os.makedirs(saves_path, exist_ok=True)

        parser_wp = Parser(file)
        dials_pairs, emb_dict = parser_wp.load_data()
        Parser.save_dials(dials_pairs)
        Parser.save_emb_dict(saves_path, emb_dict)
        train_data = parser_wp.encode_phrase_pairs(dials_pairs, emb_dict)
        rand = np.random.RandomState(parser.SHUFFLE_SEED)
        rand.shuffle(train_data)
        logging.info("Training data converted, got %d samples", len(train_data))
        train_data, test_data = parser_wp.split_train_test(train_data)
        logging.info(
            "Train set has %d phrases, test %d", len(train_data), len(test_data)
        )
        train_data = Parser.group_train_data(train_data)
        test_data = Parser.group_train_data(test_data)

        net = model.PhraseModel(
            emb_size=model.EMBEDDING_DIM,
            dict_size=len(emb_dict),
            hid_size=model.HIDDEN_STATE_SIZE,
        ).to(DEVICE)

        logging.info("Model: %s", net)

        optimizerws = OptimizerRL(
            name=name,
            net=net,
            device=DEVICE,
            emb_dict=emb_dict,
            model_base=base_model,
            train_data=train_data,
            test_data=test_data,
        )
        optimizerws.optimize()

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
