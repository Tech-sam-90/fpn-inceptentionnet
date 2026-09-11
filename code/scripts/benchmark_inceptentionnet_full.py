import sys
from pathlib import Path
REPO_ROOT = Path("/lustre07/scratch/joyinola/fpn_mamba/repo")
sys.path.insert(0, str(REPO_ROOT))

import time
import torch
from code.src.models.inceptentionnet import InceptentionNet

BATCH, N_WARMUP, N_ITERS = 4, 5, 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

for img_size in [256, 512]:
    model = InceptentionNet(stem_channels=64, branch_channels=64, num_heads=4, dropout=0.3).to(device).train()
    x = torch.randn(BATCH, 3, img_size, img_size, device=device)
    y = torch.randint(0, 2, (BATCH,), device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(N_WARMUP):
        optimizer.zero_grad()
        loss = loss_fn(model(x).reshape(-1), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        optimizer.zero_grad()
        loss = loss_fn(model(x).reshape(-1), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    step_ms = 1000 * elapsed / N_ITERS
    throughput = BATCH * N_ITERS / elapsed
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    total = sum(p.numel() for p in model.parameters())
    print(f"inceptentionnet @ {img_size}px: {step_ms:.1f} ms/step  {throughput:.1f} samples/s  "
          f"peak_mem={peak_mem_gb:.2f}GB  params={total:,}", flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
