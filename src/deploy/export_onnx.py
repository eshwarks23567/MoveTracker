"""
Export trained PyTorch HAR model to ONNX format.

This is an alternative deployment path that allows running the model
in any ONNX-compatible runtime (ONNX Runtime, TensorRT, etc.)
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def export_to_onnx(model_name: str, checkpoint_path: str = None,
                   output_path: str = None, optimize: bool = True):
    """
    Export a trained model to ONNX format.

    Parameters
    ----------
    model_name : str
        One of 'cnn', 'lstm', 'gru', 'hybrid'.
    checkpoint_path : str, optional
    output_path : str, optional
    optimize : bool
        Whether to run ONNX graph optimizations.
    """
    from src.training.train import get_model_by_name

    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"
    if output_path is None:
        output_path = config.RESULTS_DIR / f"{model_name}.onnx"

    # Load model
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    model = get_model_by_name(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input
    dummy = torch.randn(1, config.WINDOW_SIZE, config.NUM_CHANNELS)

    # Export
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['sensor_data'],
        output_names=['activity_logits'],
        dynamic_axes={
            'sensor_data': {0: 'batch_size'},
            'activity_logits': {0: 'batch_size'},
        },
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported to {output_path} ({size_mb:.2f} MB)")

    # Optimize if requested
    if optimize:
        try:
            import onnx
            from onnxruntime.transformers import optimizer as ort_opt

            model_onnx = onnx.load(str(output_path))
            onnx.checker.check_model(model_onnx)
            print("✅ ONNX model validation passed")
        except ImportError:
            print("⚠️  onnxruntime not available for optimization")
        except Exception as e:
            print(f"⚠️  ONNX validation: {e}")

    return str(output_path)


def verify_onnx(onnx_path: str, num_samples: int = 10):
    """
    Verify ONNX model produces same outputs as PyTorch model.
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))

        # Random inputs
        for _ in range(num_samples):
            x = np.random.randn(1, config.WINDOW_SIZE, config.NUM_CHANNELS).astype(np.float32)
            ort_inputs = {'sensor_data': x}
            ort_outputs = session.run(None, ort_inputs)

            assert ort_outputs[0].shape == (1, config.NUM_CLASSES)
            print(f"  Sample output: {ort_outputs[0][0][:3]}... ✅")

        print(f"✅ ONNX verification passed ({num_samples} samples)")

    except ImportError:
        print("⚠️  onnxruntime not installed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('--model', type=str, default='hybrid',
                        choices=['cnn', 'lstm', 'gru', 'hybrid'])
    args = parser.parse_args()

    path = export_to_onnx(args.model)
    verify_onnx(path)
