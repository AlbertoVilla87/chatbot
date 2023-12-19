import os
import numpy as np
import random
import logging
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
import ptan

from whatsapp.libbots import parser
from whatsapp.libbots.parser import Parser
from whatsapp.libbots import model
from . import utils

SAVES_DIR = "saves"

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHES = 10000
SAMPLES = 4


class OptimizerRL:
    def __init__(
        self,
        name: str,
        net: model.PhraseModel,
        device: str,
        model_base: str,
        train_data: list,
        test_data: list,
        emb_dict=dict,
    ):
        """_summary_
        :param name: name of model director
        :type name: str
        :param net: model
        :type net: model.PhraseModel
        :param device: cpu/gpu
        :type device: str
        :param model_base: previous model to KT
        :type model_base: str
        :param train_data: _description_
        :type train_data: list
        :param test_data: _description_
        :type test_data: list
        """
        self.name = name
        self.net = net
        self.device = device
        self.saves_path = os.path.join(SAVES_DIR, name)
        self.train_data = train_data
        self.test_data = test_data
        self.model_base = model_base
        self.emb_dict = emb_dict

    def optimize(self):
        """_summary_"""
        writer = SummaryWriter(comment="-" + self.name)
        self.net.load_state_dict(torch.load(self.model_base))
        logging.info(
            "Model loaded from %s, continue training in RL mode...",
            self.model_base,
        )

        # BEGIN token
        beg_token = torch.LongTensor([self.emb_dict[parser.BEGIN_TOKEN]]).to(
            self.device
        )

        rev_emb_dict = {idx: word for word, idx in self.emb_dict.items()}

        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            optimiser = optim.Adam(self.net.parameters(), lr=LEARNING_RATE, eps=1e-3)
            batch_idx = 0
            best_bleu = None
            for epoch in range(MAX_EPOCHES):
                random.shuffle(self.train_data)
                dial_shown = False

                total_samples = 0
                skipped_samples = 0
                bleus_argmax = []
                bleus_sample = []

                for batch in Parser.iterate_batches(self.train_data, BATCH_SIZE):
                    batch_idx += 1
                    optimiser.zero_grad()
                    input_seq, input_batch, output_batch = model.pack_batch_no_out(
                        batch, self.net.emb, self.device
                    )
                    enc = self.net.encode(input_seq)

                    net_policies = []
                    net_actions = []
                    net_advantages = []
                    beg_embedding = self.net.emb(beg_token)

                    for idx, inp_idx in enumerate(input_batch):
                        total_samples += 1
                        ref_indices = [indices[1:] for indices in output_batch[idx]]
                        item_enc = self.net.get_encoded_item(enc, idx)
                        r_argmax, actions = self.net.decode_chain_argmax(
                            item_enc,
                            beg_embedding,
                            parser.MAX_TOKENS,
                            stop_at_token=self.emb_dict[parser.END_TOKEN],
                        )
                        argmax_bleu = utils.calc_bleu_many(actions, ref_indices)
                        bleus_argmax.append(argmax_bleu)

                        if argmax_bleu > 0.99:
                            skipped_samples += 1
                            continue

                        if not dial_shown:
                            logging.info(
                                "Input: %s",
                                utils.untokenize(
                                    Parser.decode_words(inp_idx, rev_emb_dict)
                                ),
                            )
                            ref_words = [
                                utils.untokenize(Parser.decode_words(ref, rev_emb_dict))
                                for ref in ref_indices
                            ]
                            logging.info("Refer: %s", " ~~|~~ ".join(ref_words))
                            logging.info(
                                "Argmax: %s, bleu=%.4f",
                                utils.untokenize(
                                    Parser.decode_words(actions, rev_emb_dict)
                                ),
                                argmax_bleu,
                            )

                        for _ in range(SAMPLES):
                            r_sample, actions = self.net.decode_chain_sampling(
                                item_enc,
                                beg_embedding,
                                parser.MAX_TOKENS,
                                stop_at_token=self.emb_dict[parser.END_TOKEN],
                            )
                            sample_bleu = utils.calc_bleu_many(actions, ref_indices)

                            if not dial_shown:
                                logging.info(
                                    "Sample: %s, bleu=%.4f",
                                    utils.untokenize(
                                        Parser.decode_words(actions, rev_emb_dict)
                                    ),
                                    sample_bleu,
                                )

                            net_policies.append(r_sample)
                            net_actions.extend(actions)
                            net_advantages.extend(
                                [sample_bleu - argmax_bleu] * len(actions)
                            )
                            bleus_sample.append(sample_bleu)
                        dial_shown = True

                    if not net_policies:
                        continue

                    policies_v = torch.cat(net_policies)
                    actions_t = torch.LongTensor(net_actions).to(self.device)
                    adv_v = torch.FloatTensor(net_advantages).to(self.device)
                    log_prob_v = F.log_softmax(policies_v, dim=1)
                    log_prob_actions_v = (
                        adv_v * log_prob_v[range(len(net_actions)), actions_t]
                    )
                    loss_policy_v = -log_prob_actions_v.mean()

                    loss_v = loss_policy_v
                    loss_v.backward()
                    optimiser.step()

                    tb_tracker.track("advantage", adv_v, batch_idx)
                    tb_tracker.track("loss_policy", loss_policy_v, batch_idx)
                    tb_tracker.track("loss_total", loss_v, batch_idx)

                bleu_test = self.run_test()
                bleu = np.mean(bleus_argmax)
                writer.add_scalar("bleu_test", bleu_test, batch_idx)
                writer.add_scalar("bleu_argmax", bleu, batch_idx)
                writer.add_scalar("bleu_sample", np.mean(bleus_sample), batch_idx)
                writer.add_scalar(
                    "skipped_samples", skipped_samples / total_samples, batch_idx
                )
                writer.add_scalar("epoch", batch_idx, epoch)
                logging.info("Epoch %d, test BLEU: %.3f", epoch, bleu_test)
                if best_bleu is None or best_bleu < bleu_test:
                    best_bleu = bleu_test
                    logging.info("Best bleu updated: %.4f", bleu_test)
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(
                            self.saves_path, "bleu_%.3f_%02d.dat" % (bleu_test, epoch)
                        ),
                    )
                if epoch % 10 == 0:
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(
                            self.saves_path,
                            "epoch_%03d_%.3f_%.3f.dat" % (epoch, bleu, bleu_test),
                        ),
                    )

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
            ref_indices = [indices[1:] for indices in p2]
            bleu_sum += utils.calc_bleu_many(tokens, ref_indices)
            bleu_count += 1
        return bleu_sum / bleu_count
