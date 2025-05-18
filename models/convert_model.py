# convert_pong_model.py
import torch
import torch.onnx
import os

# Assuming model.py (containing QNetwork definition) is in the same directory
from model import QNetwork

# --- Configuration ---
# Path to the fine-tuned PyTorch model file you want to convert
MODEL_PATH = "../pong-medium.pth"
# The hidden dimension used when training this model (MUST match!)
HIDDEN_DIM = 128
# State dimension for Pong environment
STATE_DIM = 5
# Action dimension for Pong environment
ACTION_DIM = 3
# Output path for the ONNX model
ONNX_OUTPUT_PATH = "../dqn-mid.onnx"
# --- End Configuration ---

def convert_model():
    """Loads the PyTorch model and exports it to ONNX format."""

    print(f"Loading PyTorch model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Initialize the network structure
    # Ensure the parameters match the saved model's architecture
    qnet = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)

    # Load the trained weights (loading onto CPU for export)
    try:
        qnet.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Check if HIDDEN_DIM matches the saved model.")
        return

    # Set the model to evaluation mode (important!)
    qnet.eval()
    print("Model loaded successfully.")

    # Create a dummy input tensor with the correct shape
    # Batch size of 1, state dimension
    dummy_input = torch.randn(1, STATE_DIM, requires_grad=False)
    print(f"Using dummy input shape: {dummy_input.shape}")

    print(f"Exporting model to ONNX format at: {ONNX_OUTPUT_PATH}")
    try:
        # Export the model
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
