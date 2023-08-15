import numpy as np
import yaml

import torch

from data_loader.data_loader import DataLoader
from models.sgat_transformer.sgat_transformer import SGATTransformer
from test import test
from train import train
from utils.data_utils import create_lookup_index
from utils.logger import logger
from utils.masked_mae_loss import Masked_MAE_Loss


def train_validate(model, configs: dict, data_loader: DataLoader):
    if configs['load_saved_model']:
        model.load_state_dict(torch.load(configs['model_input_path']))

    # mse_loss_fn = nn.L1Loss()
    mse_loss_fn = Masked_MAE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=15, T_mult=1,
                                                                        eta_min=0.00005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.75)
    optimizer.zero_grad()

    min_val_loss = np.inf

    for epoch in range(configs['epochs']):
        logger.info(f"LR: {lr_scheduler.get_last_lr()}")

        dec_offset = configs['transformer']['decoder']['seq_offset']
        mae_train_loss, rmse_train_loss, mape_train_loss = train(model=model,
                                                                 data_loader=data_loader,
                                                                 optimizer=optimizer,
                                                                 loss_fn=mse_loss_fn,
                                                                 device=configs['device'],
                                                                 seq_offset=dec_offset)

        mae_val_loss, rmse_val_loss, mape_val_loss = test(_type='test',
                                                          model=model,
                                                          data_loader=data_loader,
                                                          device=configs['device'],
                                                          seq_offset=dec_offset)
        lr_scheduler.step()

        out_txt = f"Epoch: {epoch} | mae_train_loss: {mae_train_loss} | rmse_train_loss: {rmse_train_loss} " \
                  f"| mape_train_loss: {mape_train_loss} | mae_val_loss: {mae_val_loss} " \
                  f"| rmse_val_loss: {rmse_val_loss} | mape_val_loss: {mape_val_loss}"
        logger.info(out_txt)

        if min_val_loss > mae_val_loss:
            min_val_loss = mae_val_loss
            logger.info('Saving Model...')
            best_model_path = configs['model_output_path'].format(str(epoch))
            torch.save(model.state_dict(), best_model_path)  # saving model

    # testing model
    logger.info('Testing model...')
    model.load_state_dict(torch.load(best_model_path))
    mae_test_loss, rmse_test_loss, mape_test_loss = test(_type='test',
                                                         model=model,
                                                         data_loader=data_loader,
                                                         device=model_configs['device'],
                                                         seq_offset=model_configs['dec_seq_offset'])

    logger.info(f"mae_test_loss: {mae_test_loss} | rmse_test_loss: {rmse_test_loss} | mape_test_loss: {mape_test_loss}")


def prepare_data(model_configs: dict, data_configs: dict):
    data_configs['batch_size'] = model_configs['batch_size']
    data_configs['enc_features'] = model_configs['transformer']['encoder']['features']
    data_configs['dec_seq_offset'] = model_configs['transformer']['decoder']['seq_offset']
    dec_seq_len = model_configs['transformer']['decoder']['seq_len']
    enc_seq_len = model_configs['transformer']['encoder']['seq_len']

    data_loader = DataLoader(data_configs)
    data_loader.load_node_data_file()
    data_loader.load_edge_data_file()
    # data_loader.load_semantic_edge_data_file()

    max_lkup_len_enc, lkup_idx_enc, max_lkup_len_dec, lkup_idx_dec = create_lookup_index(data_configs['last_week'],
                                                                                         data_configs['last_day'],
                                                                                         data_configs['dec_seq_offset'],
                                                                                         dec_seq_len)

    model_configs['transformer']['decoder']['lookup_idx'] = lkup_idx_dec
    model_configs['transformer']['decoder']['max_lookup_len'] = max_lkup_len_dec if max_lkup_len_dec else dec_seq_len
    model_configs['transformer']['encoder']['lookup_idx'] = lkup_idx_enc
    model_configs['transformer']['encoder']['max_lookup_len'] = max_lkup_len_enc if max_lkup_len_enc else enc_seq_len

    return data_loader, model_configs


if __name__ == '__main__':
    # load configs
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

    model_configs = configs['model']
    data_configs = configs['data']
    data_loader, model_configs = prepare_data(model_configs, data_configs)

    model = SGATTransformer(configs=model_configs).to(model_configs['device'])
    train_validate(model, model_configs, data_loader)
