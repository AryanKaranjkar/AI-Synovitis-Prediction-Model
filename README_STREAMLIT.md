# Streamlit App (Standalone Inference)

A minimal UI to run the Synovitis classifier with your trained weights.

## Files to put in this folder

```
inference/
├── app.py                 # this Streamlit app
├── model.pt               # copy from your best run (e.g., runs/study2/best_overall.pt)
├── model_info.json        # optional: { "model_name": "<timm name>", "num_classes": 2 }
├── requirements.txt       # add `streamlit` here (see below)
```

### Requirements
Append to your existing `requirements.txt` or create one:
```
torch>=2.0.0
timm>=0.9.0
Pillow
torchvision
streamlit
```

## Run the app

```bash
streamlit run app.py
```

Open the URL that Streamlit prints (usually http://localhost:8501).

## How it works
- Loads your weights (either a plain `state_dict` or a dict containing `{"model": state_dict}`).
- Architecture defaults to `efficientnetv2_rw_m.agc_in1k` (you can override in the sidebar).
- Pads to white square, resizes to the chosen size (default 512), normalizes with ImageNet stats.
- Optional **TTA** averages predictions over flips.
- Shows per-class probabilities and lets you download a JSON with the results.
