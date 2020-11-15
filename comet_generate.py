"""
Code used to generate emotional reactions for sampled stories during RL tarining (Rl-Em).
"""


import sys
#sys.path.append('../comet-commonsense')


import os
import sys
import torch

#sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
from find_x_o_appx import *


dev = 1
model_file = "comet-commonsense/pretrained_models/atomic_pretrained_model.pickle"
#sampling_algorithm = "topk-1"
#category = "xReact"
opt, state_dict = interactive.load_model_file(model_file)

data_loader, text_encoder = interactive.load_data("atomic", opt)

n_ctx = data_loader.max_event + data_loader.max_effect
n_vocab = len(text_encoder.encoder) + n_ctx
model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

if dev != "cpu":
    cfg.device = int(dev)
    cfg.do_gpu = True
    torch.cuda.set_device(cfg.device)
    model.cuda(cfg.device)
else:
    cfg.device = "cpu"


def get_comet_prediction(all_story):
    # all_story list (bs) of list (5)
    all_outputs = []
    for i, story in enumerate(all_story):
        output = []
        if len(story) < 3:
            all_outputs.append(output)
            continue
        protagonist = find_gender(story)
        pos = create_pos(story, protagonist)
        for k, input_event in enumerate(story):
            #print(input_event)
            input_event = input_event.replace(",","").strip(".").strip("!").strip()
            input_event = ' '.join(i for i in input_event.split()[:17])
            category = 'xReact' if pos[k] == 'x' else 'oReact'
            sampling_algorithm = 'topk-1' if pos[k] == 'x' else 'topk-2'
            sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
            out = interactive.get_atomic_sequence(
                    input_event, model, sampler, data_loader, text_encoder, category)
            # out_file.write("%s\t%s\n" % (input_event, out[category]["beams"][0]))
            if category == "oReact" and out[category]["beams"][0] == "none":
                output.append(out[category]["beams"][1])
            else:
                output.append(out[category]["beams"][0])
        all_outputs.append(output)
    return all_outputs # list(bs) of list(5)

