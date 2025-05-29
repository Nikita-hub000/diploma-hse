"""
Экспорт обученной policy-сети (логиты действий) в ONNX.
ONNX-выход:  'action_logits' (shape [1,3]) — JS просто ищет argmax.
"""
import os, torch, torch.onnx
from reinforce_agent import PolicyValueNet   
MODEL_PATH = "../../policy/reinforce_medium.pth"
ONNX_OUT   = "../../policy_mid.onnx"
STATE_DIM  = 5
ACTION_DIM = 3
HIDDEN     = 128

def export():
    if not os.path.exists(MODEL_PATH):
        print("Trained model not found.")
        return
    net = PolicyValueNet(STATE_DIM, ACTION_DIM, HIDDEN)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()

    dummy = torch.randn(1, STATE_DIM)
    torch.onnx.export(
        net,
        dummy,
        ONNX_OUT,
        export_params=True,
        opset_version=11,
        input_names=['input_state'],
        output_names=['action_logits', 'state_value'],
        dynamic_axes={'input_state': {0: 'batch'}, 'action_logits': {0: 'batch'}, 'state_value': {0:'batch'}}
    )
    print(f"Exported to {ONNX_OUT}")

if __name__ == "__main__":
    export()

