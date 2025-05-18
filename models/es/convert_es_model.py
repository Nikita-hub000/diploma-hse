"""
Конвертация Evolution-Strategies модели в ONNX.
Выход: 'action_logits' (shape [1,3]) — в браузере берём argmax.
"""
import os
import torch
import torch.onnx
from es_agent import ESAgent          # обёртка с self.net внутри

# ── конфиг ───────────────────────────────────────────────────────
MODEL_PATH = "../../pong_es-hard.pth"   # .pth, сохранённый в train_es.py
ONNX_OUT   = "../../es-hard.onnx"
STATE_DIM  = 5
ACTION_DIM = 3
HIDDEN     = 128
# ─────────────────────────────────────────────────────────────────

def export():
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Trained model not found at {MODEL_PATH}")
        return

    # 1) создаём агент и загружаем веса
    agent = ESAgent(STATE_DIM, ACTION_DIM, HIDDEN)
    agent.load(MODEL_PATH)                 # <- у обёртки есть .load()
    net = agent.net                        # сама NN-сеть

    net.eval()
    dummy = torch.randn(1, STATE_DIM)

    # 2) экспорт
    torch.onnx.export(
        net,
        dummy,
        ONNX_OUT,
        export_params=True,
        opset_version=11,
        input_names=['input_state'],
        output_names=['action_logits'],
        dynamic_axes={'input_state': {0: 'batch'},
                      'action_logits': {0: 'batch'}}
    )
    print(f"✅ Exported to {ONNX_OUT}")

if __name__ == "__main__":
    export()
