"""
Export trained PyTorch HAR models to TensorFlow Lite for mobile deployment.

Pipeline:
    PyTorch (.pt) → ONNX (.onnx) → TF SavedModel → TFLite (.tflite)

Supports:
- Post-training INT8 quantization
- Float16 quantization
- Dynamic range quantization
- Model size and latency benchmarking
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def load_pytorch_model(model_name: str, checkpoint_path: str = None):
    """
    Load a trained PyTorch model from checkpoint.

    Parameters
    ----------
    model_name : str
        One of 'cnn', 'lstm', 'gru', 'hybrid'.
    checkpoint_path : str, optional
        Path to .pt checkpoint file.

    Returns
    -------
    model : nn.Module (in eval mode)
    """
    from src.training.train import get_model_by_name

    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = get_model_by_name(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded {model_name} model from {checkpoint_path}")
    return model


def export_to_onnx(model, model_name: str, output_path: str = None,
                   window_size: int = None, num_channels: int = None):
    """
    Export PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
    model_name : str
    output_path : str, optional
    """
    window_size = window_size or config.WINDOW_SIZE
    num_channels = num_channels or config.NUM_CHANNELS

    if output_path is None:
        output_path = config.RESULTS_DIR / f"{model_name}.onnx"

    dummy_input = torch.randn(1, window_size, num_channels)

    torch.onnx.export(
        model,
        dummy_input,
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

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported ONNX model: {output_path} ({file_size:.2f} MB)")
    return str(output_path)


def convert_onnx_to_tflite(onnx_path: str, output_path: str = None,
                           quantize: str = 'dynamic'):
    """
    Convert ONNX model to TensorFlow Lite.

    Parameters
    ----------
    onnx_path : str
        Path to .onnx file.
    output_path : str, optional
        Path for .tflite output.
    quantize : str
        Quantization mode: 'none', 'dynamic', 'float16', 'int8'.
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install: pip install onnx onnx-tf tensorflow")
        return None

    if output_path is None:
        output_path = str(config.TFLITE_MODEL_PATH)

    # ONNX → TF SavedModel
    print("  Converting ONNX → TF SavedModel...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    savedmodel_dir = str(Path(output_path).parent / "tf_savedmodel")
    tf_rep.export_graph(savedmodel_dir)

    # TF SavedModel → TFLite
    print(f"  Converting TF SavedModel → TFLite (quantize={quantize})...")
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)

    # Apply quantization
    if quantize == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        # Representative dataset for full INT8 quantization
        def representative_dataset():
            X = np.load(config.PROCESSED_DATA_DIR / "windows_X.npy")
            for i in range(min(100, len(X))):
                yield [X[i:i+1].astype(np.float32)]
        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ TFLite model saved: {output_path} ({file_size:.2f} MB)")

    return output_path


def export_direct_tflite(model, model_name: str, output_path: str = None,
                          window_size: int = None, num_channels: int = None,
                          quantize: str = 'dynamic'):
    """
    Alternative: Export directly using TF Lite via tracing.
    Falls back to ONNX pipeline if direct conversion fails.
    """
    try:
        import tensorflow as tf

        window_size = window_size or config.WINDOW_SIZE
        num_channels = num_channels or config.NUM_CHANNELS

        if output_path is None:
            output_path = str(config.TFLITE_MODEL_PATH)

        # Trace PyTorch model
        model.eval()
        dummy = torch.randn(1, window_size, num_channels)

        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)

        # Get TF-compatible weights by running through ONNX
        onnx_path = export_to_onnx(model, model_name)
        return convert_onnx_to_tflite(onnx_path, output_path, quantize)

    except Exception as e:
        print(f"⚠️  Direct conversion failed: {e}")
        print("   Falling back to ONNX pipeline...")
        onnx_path = export_to_onnx(model, model_name)
        return convert_onnx_to_tflite(onnx_path, output_path, quantize)


def benchmark_tflite(tflite_path: str, num_runs: int = 100):
    """
    Benchmark TFLite model inference speed.

    Parameters
    ----------
    tflite_path : str
    num_runs : int

    Returns
    -------
    dict with avg_ms, std_ms, file_size_mb
    """
    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        file_size = os.path.getsize(tflite_path) / (1024 * 1024)

        results = {
            'avg_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'file_size_mb': file_size,
            'input_shape': list(input_shape),
            'meets_latency_target': np.mean(times) < config.TARGET_INFERENCE_MS,
            'meets_size_target': file_size < config.TARGET_MODEL_SIZE_MB,
        }

        print(f"\n📊 TFLite Benchmark ({num_runs} runs):")
        print(f"   Avg inference: {results['avg_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"   Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
        print(f"   Model size: {results['file_size_mb']:.2f} MB")
        print(f"   Latency target (<{config.TARGET_INFERENCE_MS}ms): "
              f"{'✅' if results['meets_latency_target'] else '❌'}")
        print(f"   Size target (<{config.TARGET_MODEL_SIZE_MB}MB): "
              f"{'✅' if results['meets_size_target'] else '❌'}")

        return results

    except ImportError:
        print("❌ TensorFlow not available for benchmarking")
        return None


def full_export_pipeline(model_name: str, quantize: str = 'dynamic'):
    """
    Run the complete export pipeline:
    Load model → ONNX → TFLite → Benchmark
    """
    print(f"\n{'=' * 60}")
    print(f"Exporting {model_name.upper()} to TFLite")
    print(f"{'=' * 60}")

    # Load model
    model = load_pytorch_model(model_name)

    # Export to ONNX
    onnx_path = export_to_onnx(model, model_name)

    # Convert to TFLite
    tflite_path = convert_onnx_to_tflite(onnx_path, quantize=quantize)

    # Benchmark
    if tflite_path:
        benchmark_tflite(tflite_path)

    return tflite_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export HAR model to TFLite")
    parser.add_argument('--model', type=str, default='hybrid',
                        choices=['cnn', 'lstm', 'gru', 'hybrid'])
    parser.add_argument('--quantize', type=str, default='dynamic',
                        choices=['none', 'dynamic', 'float16', 'int8'])
    parser.add_argument('--input', type=str, default=None,
                        help='Path to .pt checkpoint')

    args = parser.parse_args()
    full_export_pipeline(args.model, args.quantize)
