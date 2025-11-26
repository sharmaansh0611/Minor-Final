#!/usr/bin/env python3
"""
tinyml_infer.py

Simple one-shot TinyML worker:
 - Reads JSON from stdin with keys: model_path and features (list of floats)
 - Attempts to load Keras model and predict
 - Prints JSON {"pred": 0.37} to stdout on success
 - If any error occurs, exits with non-zero status and writes error to stderr

This script is intentionally one-shot (load -> predict -> exit) so the Streamlit
process can call it with a timeout and recover if it blocks.
"""
import sys
import json
import os
import traceback

def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            print(json.dumps({"error": "no input"}))
            sys.exit(2)
        payload = json.loads(raw)
        model_path = payload.get("model_path")
        features = payload.get("features")
        if model_path is None or features is None:
            print(json.dumps({"error": "missing model_path or features"}))
            sys.exit(2)

        # Try to limit native threads (may help with lock contention)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        # Import tensorflow lazily
        import numpy as _np
        try:
            import tensorflow as tf
        except Exception as e:
            # some systems use tflite-runtime or different TF packaging
            print(json.dumps({"error": f"tensorflow import failed: {e}"}))
            sys.exit(3)

        # Load model (this may take a few seconds)
        model = tf.keras.models.load_model(model_path)

        # Prepare features (ensure shape [1, n])
        arr = _np.array(features, dtype=_np.float32).reshape(1, -1)
        pred = model.predict(arr, verbose=0)
        # convert to float
        p = float(pred.ravel()[0])
        print(json.dumps({"pred": p}))
        sys.exit(0)
    except Exception as e:
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        print(json.dumps({"error": str(e)}))
        sys.exit(4)

if __name__ == "__main__":
    main()
