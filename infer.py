from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test import test_single_video
import option
import time


if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/wsanodet_mix2.pkl').items()})
    st = time.time()
    message, message_second, message_frames  =  test_single_video(test_loader, model, device)
    time_elapsed = time.time() - st
    print(message + message_frames)
    print(message + message_second)
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
