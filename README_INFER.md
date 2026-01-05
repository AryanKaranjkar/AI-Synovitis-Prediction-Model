# Inference (Standalone)

Minimal drop-in folder to run the Synovitis binary classifier.

## Files to place here

```
inference/
├── infer_min.py          # this script
├── model.pt              # copy from your training run (e.g., runs/study2/best_overall.pt)
├── requirements.txt      # minimal deps for inference
```

- The model is hard-coded as: `efficientnetv2_rw_m.agc_in1k` with `num_classes=2`.
- Preprocessing: square pad to **white**, resize to **512×512**, ImageNet normalization.

## Install

```bash
pip install -r requirements.txt
```

## Run

Single image:
```bash
python infer_min.py --model model.pt --img example.jpg
```

With simple TTA (averages over flips):
```bash
python infer_min.py --model model.pt --img example.jpg --tta
```

Save JSON output:
```bash
python infer_min.py --model model.pt --img example.jpg --out_json result.json
```

## Notes

- If your model was trained on a different architecture or number of classes, edit `infer_min.py`:
  ```python
  model = timm.create_model("<your-arch>", pretrained=False, num_classes=<K>)
  ```
- CPU-only is supported; it’s slower. CUDA is used automatically if available.
- The weights file can be:
  - a plain `state_dict` (output of `model.state_dict()`), or
  - a dict containing `{"model": state_dict, ...}`.
