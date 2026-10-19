#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线训练：行为克隆 MLP（CPU 即可，无需 GPU）.

读取 collect_data.py 生成的 npz 数据集，以加权 MSE 训练 CruiseMLP：
    loss = mean(((v_pred - v)/max_lin)^2) + mean(((w_pred - w)/max_ang)^2)
即把线速度/角速度各自按其物理量程归一化后再求误差，避免角速度量程大
(2.84 rad/s) 而线速度量程小 (0.22 m/s) 导致的损失失衡。

产物:
    models/model.pth       网络权重（*.pth 不入库，可由本脚本复现）
    models/train_log.csv   每个 epoch 的训练/验证损失（文本，入库画损失曲线）
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

import depth_e2e_core as core

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(features, actions, n_in, hidden, max_lin, max_ang,
          epochs, batch, lr, seed, log_path):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_total = features.shape[0]
    idx = np.random.permutation(n_total)
    n_val = max(1, int(0.1 * n_total))                  # 10% 留作验证
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    x_all = torch.from_numpy(features)
    y_all = torch.from_numpy(actions)

    model = core.CruiseMLP(n_in=n_in, hidden=hidden, max_lin=max_lin, max_ang=max_ang)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    log_rows = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(tr_idx.numel())
        losses = []
        for k in range(0, perm.numel(), batch):
            sel = tr_idx[perm[k:k + batch]]
            x, y = x_all[sel], y_all[sel]
            opt.zero_grad()
            out = model(x)                               # (B, 2) 物理值
            loss = (((out[:, 0] - y[:, 0]) / max_lin) ** 2).mean() \
                 + (((out[:, 1] - y[:, 1]) / max_ang) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        train_loss = sum(losses) / max(1, len(losses))

        model.eval()
        with torch.no_grad():
            out_v = model(x_all[val_idx])
            val_loss = float((((out_v[:, 0] - y_all[val_idx, 0]) / max_lin) ** 2).mean()
                             + (((out_v[:, 1] - y_all[val_idx, 1]) / max_ang) ** 2).mean())
        log_rows.append((epoch, train_loss, val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print("epoch %4d | train %.5f | val %.5f" % (epoch, train_loss, val_loss))

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for row in log_rows:
            writer.writerow(["%d" % row[0], "%.6f" % row[1], "%.6f" % row[2]])

    model.load_state_dict(best_state)
    return model, best_val


def main():
    ap = argparse.ArgumentParser(description="行为克隆 MLP 训练")
    ap.add_argument("--data", default="data/depth_e2e_dataset.npz",
                    help="数据集 npz 路径（相对包根或绝对路径）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-lin", type=float, default=0.22)
    ap.add_argument("--max-ang", type=float, default=2.84)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models", help="权重与训练日志输出目录（相对包根）")
    args = ap.parse_args()

    data_path = args.data if os.path.isabs(args.data) else os.path.join(PKG_ROOT, args.data)
    if not os.path.isfile(data_path):
        raise SystemExit("未找到数据集 %s，请先用 main.launch mode:=collect 采集。" % data_path)
    data = np.load(data_path)
    features, actions = data["features"], data["actions"]
    print("数据集: %s | 样本 %d | 特征维度 %d" % (data_path, features.shape[0], features.shape[1]))

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(PKG_ROOT, args.out)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    log_path = os.path.join(out_dir, "train_log.csv")

    model, best_val = train(features, actions, features.shape[1], args.hidden,
                            args.max_lin, args.max_ang,
                            args.epochs, args.batch, args.lr, args.seed, log_path)

    model_path = os.path.join(out_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print("训练完成: best val loss %.5f" % best_val)
    print("权重已保存: %s（不入库，部署时与 config 中 model/path 对应）" % model_path)
    print("训练曲线数据: %s（入库，文档据此绘制损失曲线）" % log_path)


if __name__ == "__main__":
    main()
