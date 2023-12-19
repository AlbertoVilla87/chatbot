import os
import numpy as np
import random
import logging
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F

from whatsapp.libbots import parser
from whatsapp.libbots.parser import Parser
from whatsapp.libbots import model
from . import utils

SAVES_DIR = "saves"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHES = 101

TEACHER_PROB = 0.5


class OptimizerCE:
    def __init__(
        self,
        name: str,
        net: model.PhraseModel,
        device: str,
        train_data: list,
        test_data: list,
    ):
        self.name = name
        self.net = net
        self.device = device
        self.saves_path = os.path.join(SAVES_DIR, name)
        self.train_data = train_data
        self.test_data = test_data

    def optimize(self):
        """_summary_"""
        writer = SummaryWriter(comment="-" + self.name)
        optimiser = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        best_bleu = None
        for epoch in range(MAX_EPOCHES):
            losses = []
            bleu_sum = 0.0
            bleu_count = 0
            for batch in Parser.iterate_batches(self.train_data, BATCH_SIZE):
                optimiser.zero_grad()
                input_seq, out_seq_list, _, out_idx = model.pack_batch(
                    batch, self.net.emb, self.device
                )
                enc = self.net.encode(input_seq)

                net_results = []
                net_targets = []
                for idx, out_seq in enumerate(out_seq_list):
                    ref_indices = out_idx[idx][1:]
                    enc_item = self.net.get_encoded_item(enc, idx)
                    if random.random() < TEACHER_PROB:
                        r = self.net.decode_teacher(enc_item, out_seq)
                        bleu_sum += model.seq_bleu(r, ref_indices)
                    else:
                        r, seq = self.net.decode_chain_argmax(
                            enc_item, out_seq.data[0:1], len(ref_indices)
                        )
                        bleu_sum += utils.calc_bleu(seq, ref_indices)
                    net_results.append(r)
                    net_targets.extend(ref_indices)
                    bleu_count += 1
                results_v = torch.cat(net_results)
                targets_v = torch.LongTensor(net_targets).to(self.device)
                loss_v = F.cross_entropy(results_v, targets_v)
                loss_v.backward()
                optimiser.step()

                losses.append(loss_v.item())
            bleu = bleu_sum / bleu_count
            bleu_test = self.run_test()
            logging.info(
                "Epoch %d: mean loss %.3f, mean BLEU %.3f, test BLEU %.3f",
                epoch,
                np.mean(losses),
                bleu,
                bleu_test,
            )
            writer.add_scalar("loss", np.mean(losses), epoch)
            writer.add_scalar("bleu", bleu, epoch)
            writer.add_scalar("bleu_test", bleu_test, epoch)
            if best_bleu is None or best_bleu < bleu_test:
                if best_bleu is not None:
                    out_name = os.path.join(
                        self.saves_path, "pre_bleu_%.3f_%02d.dat" % (bleu_test, epoch)
                    )
                    torch.save(self.net.state_dict(), out_name)
                    logging.info("Best BLEU updated %.3f", bleu_test)
                best_bleu = bleu_test

            if epoch % 10 == 0:
                out_name = os.path.join(
                    self.saves_path,
                    "epoch_%03d_%.3f_%.3f.dat" % (epoch, bleu, bleu_test),
                )
                torch.save(self.net.state_dict(), out_name)

        writer.close()

    def run_test(self):
        bleu_sum = 0.0
        bleu_count = 0
        for p1, p2 in self.test_data:
            input_seq = model.pack_input(p1, self.net.emb, self.device)
            enc = self.net.encode(input_seq)
            _, tokens = self.net.decode_chain_argmax(
                enc,
                input_seq.data[0:1],
                seq_len=parser.MAX_TOKENS,
                stop_at_token=parser.END_TOKEN,
            )
            bleu_sum += utils.calc_bleu(tokens, p2[1:])
            bleu_count += 1
        return bleu_sum / bleu_count
