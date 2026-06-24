# DEGA-SDP

DEGA-SDP (Dual-Evidence Gated Adaptive Software Defect Prediction) is a lightweight software defect prediction model. It combines:

1. global discriminative evidence from a gradient boosting classifier,
2. local defect-pattern evidence from defect and clean class prototypes,
3. a reliability gate based on uncertainty, evidence agreement, and prototype density,
4. internal validation for gate strength and decision-threshold calibration.

The code supports within-project defect prediction (WPDP) and cross-project defect prediction (CPDP) experiments on Kamei and PROMISE datasets.

## Repository Structure

```text
DEGA-SDP/
├── run_experiments.py          # command line experiment runner
├── src/
│   ├── __init__.py
│   ├── config.py               # experiment constants and paper baseline values
│   ├── data.py                 # Kamei/PROMISE dataset discovery and loaders
│   ├── model.py                # DEGA-SDP model, gate fusion, metrics
│   ├── experiment_utils.py     # CSV writing and result summarization
│   └── dega_sdp.py             # public API re-exports for runner compatibility
├── results/                    # reference summaries from the final experiments
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment

Recommended Python version: 3.9 or later.

```bash
conda create -n dega-sdp python=3.10 -y
conda activate dega-sdp
pip install -r requirements.txt
```

## Dataset Download

This project expects the Kamei and PROMISE CSV files used by the Nested-Stacking defect prediction study.

Recommended source containing both datasets:

- HFS-Nested-Stacking repository: https://github.com/WangHuoShanPY/HFS-Nested-Stacking

Download it under a local data directory:

```bash
git clone https://github.com/WangHuoShanPY/HFS-Nested-Stacking.git data/HFS-Nested-Stacking
```

The runner accepts a `--data-root` argument and automatically searches these layouts:

```text
<data-root>/HFS-Nested-Stacking/PROMISE
<data-root>/HFS-Nested-Stacking/Kamei
<data-root>/PROMISE
<data-root>/Kamei
<data-root>/data
```

For example, if the repository was cloned into `data/HFS-Nested-Stacking`, use:

```bash
python run_experiments.py --data-root data --mode smoke
```

If your datasets are stored as in the original local reproduction directory, use:

```bash
python run_experiments.py --data-root D:\nested-stacking-defect-prediction --mode smoke
```

## Running Experiments

Smoke test, using a small WPDP-PROMISE subset:

```bash
python run_experiments.py --data-root data --mode smoke
```

Run one experiment group:

```bash
python run_experiments.py --data-root data --mode wpdp-kamei
python run_experiments.py --data-root data --mode wpdp-promise
python run_experiments.py --data-root data --mode cpdp-kamei
python run_experiments.py --data-root data --mode cpdp-promise6
```

Run the selected experiment setting used in the paper analysis:

```bash
python run_experiments.py --data-root data --mode selected
```

Run all supported experiments:

```bash
python run_experiments.py --data-root data --mode all
```

Generated CSV files are written to `results/` by default. You can change the output directory:

```bash
python run_experiments.py --data-root data --mode smoke --output-dir outputs
```

## Output Metrics

The result CSV contains:

- weighted F1-score,
- AUC,
- defect-class Precision,
- defect-class Recall,
- defect-class F1-score,
- selected prototype number `k`,
- selected gate correction strength `beta`,
- calibrated threshold,
- mean reliability gate value.

The summary CSV reports mean performance by dataset group and comparison against the reported Nested-Stacking and best existing baseline F1 values.

## Reference Results

The `results/` directory contains the final grouped summaries from the local experiments. These files are provided for quick inspection and do not include raw datasets.

## Notes

- This repository intentionally excludes paper drafts, paper figures, PPT files, image-generation scripts, and intermediate analysis artifacts.
- The datasets are not redistributed in this repository. Please download them from the dataset source above.
- Random seeds are fixed in the runner, but minor numeric differences may occur across scikit-learn versions.

