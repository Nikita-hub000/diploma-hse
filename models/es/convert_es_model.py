
import os
import torch
import torch.onnx
from es_agent import ESAgent         

MODEL_PATH = "../../pong_es-easy.pth"   
ONNX_OUT   = "../../es-easy.onnx"
STATE_DIM  = 5
ACTION_DIM = 3
HIDDEN     = 128

def export():
    if not os.path.isfile(MODEL_PATH):
        print(f"Trained model not found at {MODEL_PATH}")
        return

    agent = ESAgent(STATE_DIM, ACTION_DIM, HIDDEN)
    agent.load(MODEL_PATH)               
    net = agent.net                        

    net.eval()
    dummy = torch.randn(1, STATE_DIM)

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
    print(f"Exported to {ONNX_OUT}")

if __name__ == "__main__":
    export()
