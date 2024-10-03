import os
from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test import save_results, test_single_video
import option
import time


if __name__ == '__main__':
    args = option.parser.parse_args()

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == 'cpu':
        model_dict = model.load_state_dict(
             {k.replace('module.', ''): v for k, v in torch.load('ckpt/wsanodet_mix2.pkl', map_location=torch.device('cpu')).items()})
    else:
        model_dict = model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load('ckpt/wsanodet_mix2.pkl').items()})
    
    st = time.time()
    results  =  test_single_video(test_loader, model, device, args)
    save_results(results, os.path.join(args.output_path, 'results.npy'))

