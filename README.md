# LULC Semantic Segmentation from Sentinel-1 SAR Data

Deep Learning course project. M.Sc. Applied Data Science and Analytics, SRH Hochschule Heidelberg. Supervisor: Prof. Dr. Binh Vu.

**Author:** Sanjeev Veeramani

Land use / land cover (LULC) semantic segmentation using **Sentinel-1 SAR only** (no optical imagery), across 9 classes: Tree cover, Shrubland, Grassland, Cropland, Built-up, Bare/sparse vegetation, Permanent water, Herbaceous wetland, Mangroves. The project collects and preprocesses a 32-city Sentinel-1 dataset, trains and compares 8 segmentation architectures across 4 encoder sources, and packages the winning model into an interactive prototype that runs on live satellite data and cross-checks its output against an independent, optical-based reference.

---

## Evaluation Protocol

- **Data split:** 32 cities total, 20 train / 1 validation / 2 same-region test / 3 similar-unseen test / 4 dissimilar test, 256×256 px patches, no city appearing in more than one split.
- **Labels:** ESA WorldCover, remapped to the 9 project classes.
- **Metric:** mean Intersection over Union (mIoU), computed on the real held-out test splits, not validation data.
- **Model selection:** 8 architectures × 4 encoder sources trained and compared; the winner was chosen by real test-set performance.
- **External validation:** Google Dynamic World, independent of both the SAR input and the WorldCover training labels, used as a case-study spot-check in the prototype, not used for training or model selection.

---

## Key Findings

Real test-set mIoU (mean Intersection over Union), evaluated across three held-out test groups, each progressively less similar to the training cities:

| Test split | Cities | mIoU |
|---|---|---|
| Same-region | 2 (unseen, same region as training) | 0.45 |
| Similar-unseen | 3 (unseen, broadly similar climate/region) | 0.32 |
| Dissimilar | 4 (unseen, different region/climate entirely) | 0.23 |

Winning model: **UnetPlusPlus**, ResNet50 encoder, SAR-pretrained (SSL4EO MoCo weights), the best of 8 architectures × 4 encoder sources compared, selected on real test-set performance rather than validation score alone.

Scores in the 0.23–0.45 range are consistent with what SAR-only (no optical) segmentation work generally reports; radar backscatter alone carries less distinguishing information than optical imagery, particularly between visually different but radar-similar classes such as grassland, cropland, and shrubland.

**Case-study cross-check** (prototype, `04_prototype_webapp.ipynb`): model predictions for a chosen location and month are compared pixel-by-pixel against Google Dynamic World, an independently published, optical-based classifier never used in training or label generation. Pixel-agreement in example runs ranged from roughly 56% to 83% depending on location, with agreement consistently stronger on dominant classes (Built-up, Permanent water) than on smaller, less-represented ones. This is a qualitative spot-check, not a statistical benchmark, SAR and optical sensors pick up different physical signals and are not expected to match exactly.

---

## What the Experiments Showed

A follow-up tuning pass on the winning architecture (4 training variants: extended training, higher Dice-loss weighting, lower learning rate, and speckle-noise augmentation, each evaluated with test-time augmentation and ensembled together) produced no meaningful net improvement: +0.00004 mIoU on same-region, +0.013 on similar-unseen, -0.006 on dissimilar. Averaged across all three splits, the change is within normal run-to-run noise. The original single model's numbers (0.45 / 0.32 / 0.23) are reported above as the final result rather than the ensemble, since the ensemble did not actually outperform it.

Two factors limit how high mIoU can realistically go here, independent of training choices:
- **Label noise.** Labels come from ESA WorldCover, not hand-verified ground truth; WorldCover itself has known weak spots for some classes (e.g. Shrubland), so some model "errors" are disagreements with an imperfect label source rather than genuine misclassification.
- **Rare classes.** Classes that appear in very few training patches (Shrubland, Mangroves) have too little data to learn well, pulling down the averaged mIoU even where common classes score well.

The decline from 0.45 → 0.23 across test groups is the intended outcome of the train/test city split, not a flaw: cities were deliberately grouped by similarity to training data specifically to measure this generalization gap.

---

## Execution

The repository is obtained with `git clone https://github.com/sanjeevveeramani/lulc-sentinel1-sar.git`, and dependencies are installed with `pip install -r requirements.txt`. Execution assumes the accounts described under "Required Accounts and API Keys" are in place, with the Earth Engine project ID and `CKPT_DIR` path in each notebook matching the executing account, and the dataset and model weight linked under "Deliverables" placed at the paths set near the top of each notebook.

The notebooks form a pipeline, run in order: data collection (`01_data_collection.ipynb` → `01b` → `01d` → `01e`), preprocessing (`02_preprocessing_32cities_update.ipynb`), model training (`03a_model_training_main.ipynb`, followed by `03b` and `03c`, both of which depend on `03a`'s output), and evaluation (`03d_graphs_and_evaluation.ipynb` → `03e_visualizations.ipynb`). The prototype (`Prototype code/04_prototype_webapp.ipynb`) sits outside this training pipeline: it depends only on the trained model weight, so it runs on its own, directly from the pre-trained weight linked under "Deliverables", without the rest of the pipeline being run first.

---

## Requirements

- Python 3.10+
- Dependencies are listed in `requirements.txt` and installed with:
  ```bash
  pip install -r requirements.txt
  ```
- The training notebooks (`03a`–`03c`) are GPU-bound; the prototype (`04`) runs a single lightweight inference pass per month and does not require one.

## Required Accounts and API Keys

- **Google Earth Engine account**, registered to a Google Cloud project with the Earth Engine API enabled, required for any notebook that pulls Sentinel-1 or Dynamic World data (`01`, `01b`, `01d`, `01e`, `02`, `04`). Registration is done at https://earthengine.google.com; the project ID in `ee.Initialize(project='lulc-sentinel1')` is then set to the registered project's ID.
- **Google Drive access** (the same Google account as above is simplest), used to store the dataset, checkpoints, and exported patches. No separate key is needed beyond normal Google sign-in; file paths (`CKPT_DIR`, dataset folder) are set to wherever these files are placed in the executing account's Drive.
- No other third-party API keys are required; all other dependencies (PyTorch, segmentation-models-pytorch, torchgeo, Gradio, etc.) are open-source and installed via `requirements.txt`.

---

## Repository Layout

```
Training codes/
  01_data_collection.ipynb                      Sentinel-1 patch collection, main city set
  01b_data_collection_additional_cities.ipynb    Patch collection, additional cities
  01d_test_cities_update.ipynb                   Held-out test city set, update
  01e_refix_city_boxes.ipynb                     City bounding box fix
  02_preprocessing_32cities_update.ipynb         Final 32-city dataset build (speckle filter,
                                                  dB clip, normalize, patch, split)
  03a_model_training_main.ipynb                  Train + compare 8 architectures x 4 encoders
  03b_step_b_standalone.ipynb                    Continued training, winning encoder
  03c_step_c_standalone.ipynb                    SegNet / FC-DenseNet baselines
  03d_graphs_and_evaluation.ipynb                Final evaluation + comparison graphs, real test set
  03e_visualizations.ipynb                       Result plots only, no training, no GPU
Prototype code/
  04_prototype_webapp.ipynb                      Interactive Gradio prototype + case study
README.md
requirements.txt
```

---

## Deliverables

| | Link |
|---|---|
| **Dataset** | https://drive.google.com/drive/folders/1o8Exec1QDcK5hJkT6trTvJBFj5pwHg5k?usp=sharing |
| **Training code** | this repo — `Training codes/` folder — https://github.com/sanjeevveeramani/lulc-sentinel1-sar |
| **Model weight** | https://drive.google.com/drive/folders/1qVvAmWxNdT2jveXTB0qvH-9eOKo4XnlA?usp=sharing (`unetplusplus_ssl4eo_moco_best.pt`, 187.2 MB) |
| **Prototype code** | this repo — `Prototype code/04_prototype_webapp.ipynb` — https://github.com/sanjeevveeramani/lulc-sentinel1-sar |

**Setup notes:** Cell 1 of `04_prototype_webapp.ipynb` references two paths tied to the author's own Google account: `ee.Initialize(project='lulc-sentinel1')`, the author's personal Earth Engine project, which sits at whatever project ID the executing account is registered under; and `CKPT_DIR`, the author's Google Drive location for the model weight file, which sits wherever that file (linked above) is placed in the executing account's Drive.

---

## Limitations

- Accuracy drops in regions unlike the training data (confirmed by the dissimilar test cities).
- SAR backscatter alone confuses some classes with similar radar signatures (e.g. grassland/cropland/shrubland).
- Backscatter changes with season and soil moisture, which can look like real land-cover change even when nothing changed.
- Reported percentages are raw pixel counts from an unvalidated classifier; small month-to-month differences can fall within classification error, not real change.
