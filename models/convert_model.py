import torch
import torch.onnx
import os

from model import QNetwork


MODEL_PATH = "../dqn/dqn_easy.pth"
HIDDEN_DIM = 128
STATE_DIM = 5
ACTION_DIM = 3
ONNX_OUTPUT_PATH = "../dqn-easy-new.onnx"

def convert_model():
    print(f"Loading PyTorch model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return


    qnet = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)

    try:
        qnet.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Check if HIDDEN_DIM matches the saved model.")
        return

    qnet.eval()
    print("Model loaded successfully.")


    dummy_input = torch.randn(1, STATE_DIM, requires_grad=False)
    print(f"Using dummy input shape: {dummy_input.shape}")

    print(f"Exporting model to ONNX format at: {ONNX_OUTPUT_PATH}")
    try:
        torch.onnx.export(
            qnet,                                             # model being run
            dummy_input,                                      # model input
            ONNX_OUTPUT_PATH,                                 # where to save the model
            export_params=True,                               # store the trained parameter weights
            opset_version=11,                                 # the ONNX version to export the model to
            do_constant_folding=True,                         # whether to execute constant folding
            input_names = ['input_state'],                    # model's input names
            output_names = ['output_q_values'],               # model's output names
            dynamic_axes={'input_state' : {0 : 'batch_size'}, # variable length axes
                          'output_q_values' : {0 : 'batch_size'}}
        )
        print("Model successfully exported to ONNX.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == "__main__":
    convert_model()
