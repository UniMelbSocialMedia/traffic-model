import torch

from data_loader.data_loader import DataLoader
from utils.math_utils import calculate_loss


def train(model: torch.nn.Module,
          data_loader: DataLoader,
          optimizer,
          loss_fn: torch.nn.Module,
          device: str,
          seq_offset: int = 0) -> tuple:

    offset = 0
    mae_train_loss = 0.
    rmse_train_loss = 0.
    mape_train_loss = 0.

    model.train()

    for batch in range(data_loader.n_batch_train):
        train_x, train_graph_x, train_y, train_graph_y, train_y_target = data_loader.load_batch(_type='train',
                                                                                                offset=offset,
                                                                                                batch_size=data_loader.batch_size,
                                                                                                device=device)
        out = model(train_x, train_graph_x, train_y, train_graph_y, True)
        out = out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1)

        train_y_tensor = ()
        for y in train_y_target:
            y = y[seq_offset:]
            train_y_tensor = (*train_y_tensor, y[:, :, 0])
        train_y_target = torch.stack(train_y_tensor)
        train_y_target = train_y_target.view(
            train_y_target.shape[0] * train_y_target.shape[1] * train_y_target.shape[2], -1)

        loss = loss_fn(out, train_y_target)

        mae_loss_val, rmse_loss_val, mape_loss_val = calculate_loss(y_pred=out,
                                                                    y=train_y_target,
                                                                    _max=data_loader.dataset.get_max(),
                                                                    _min=data_loader.dataset.get_min())
        mae_train_loss += mae_loss_val
        rmse_train_loss += rmse_loss_val
        mape_train_loss += mape_loss_val

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        offset += data_loader.batch_size

        if offset % 100 == 0:
            mae_tmp_loss = mae_train_loss / float(batch + 1)
            rmse_tmp_loss = rmse_train_loss / float(batch + 1)
            mape_tmp_loss = mape_train_loss / float(batch + 1)
            print(f"all_batch: {data_loader.n_batch_train} | batch: {batch} | mae_tmp_loss: {mae_tmp_loss}"
                  f" | rmse_tmp_loss: {rmse_tmp_loss} | mape_tmp_loss: {mape_tmp_loss}")

    mae_train_loss = mae_train_loss / float(data_loader.n_batch_train)
    rmse_train_loss = rmse_train_loss / float(data_loader.n_batch_train)
    mape_train_loss = mape_train_loss / float(data_loader.n_batch_train)
    return mae_train_loss, rmse_train_loss, mape_train_loss
