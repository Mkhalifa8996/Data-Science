# Speckle-Noise Removal for Breast Ultrasound Images

A CNN denoiser for breast ultrasound images, trained with **knowledge
distillation**. A student network of 16 convolutional blocks learns to remove
synthetic speckle noise, and a U-Net teacher contributes an extra distillation
term to the loss.

Notebook: **`mk-model.ipynb`** (originally run on Kaggle with a GPU).

---

## Pipeline

```
Breast_Dataset/train --> augment (rotate / flip) --> clean image -->+--> save clean PNG
                                                                      +--> add speckle noise --> save noisy PNG
Breast_Dataset/test  --> (no augmentation) --------> same as above

PNG pairs --> read --> 64x64 patches --> train student --> evaluate --> visualise
```

| Step | What happens |
|---|---|
| Augmentation | `ImageDataGenerator` with rotation <= 20 degrees, horizontal and vertical flips, `fill_mode='nearest'`, and a 70/30 train/validation split. Images are resized to 256x256. |
| Noise | Multiplicative speckle: `noisy = clean + clean * N(0, sigma)`, with sigma = 0.01, clipped to [0, 1]. |
| Patching | Each 256x256 image is cut into 16 non-overlapping 64x64 patches. |
| Training | The student is trained on (noisy patch -> clean patch) pairs with a custom distillation loss. |
| Evaluation | PSNR, SSIM, CNR, ENL, EPI, MSE, RMSE, FOM and NIQE, averaged over the test patches. |

## Notebook layout

The notebook is 43 cells: 23 code cells and 20 markdown cells that document each
step. Run them in order.

| Section | Content |
|---|---|
| 1. Imports | All libraries in one cell, plus GPU memory growth |
| 2. Configuration | Every constant and path: image size, patch size, noise sigma, epochs, learning rate, loss weights, batch counts, folders, and the student's block count |
| 3. Function declarations | Noise generation, dataset helpers, patch extraction and reconstruction, metrics, plotting |
| 4. Dataset preparation | Create the folders, build the generators, then write the clean/noisy PNG pairs to disk |
| 5. Load and patch | Read the PNGs back and extract the 64x64 patches |
| 6. Model declarations | Student, teacher U-Net, custom loss |
| 7. Build and compile | Instantiate and compile both models, set up the LR callback |
| 8. Training | `model.fit` |
| 9. Evaluation | Batched metrics, the metric table, and `model.evaluate` |
| 10. Visual check | Clean vs noisy vs denoised on one test image |
| 11. Curves and summaries | Loss/accuracy plots, `summary()` for both models |

## Models

**Student (denoiser)** -- 16 x [Conv 3x3 (64) -> BatchNorm -> LeakyReLU(0.2)] ->
Dense(512) -> Conv 3x3 (3, linear).

That is 51 layers in total: the input layer, 16 blocks of 3 layers, the Dense
layer and the output convolution. In the original run this model had
**1,816,651 parameters**, of which **604,867 were trainable** (the rest are the
frozen BatchNorm statistics and the 1,209,736 Adam slots).

**Teacher (U-Net)** -- 4 down-sampling stages (64, 128, 256, 512 filters) and a
1024-filter bottleneck. Each stage is 3 x Conv -> BatchNorm -> LeakyReLU. Dropout
of 0.5 sits at the two deepest levels, with skip connections on the way up and a
sigmoid output. **50,257,347 parameters**, of which 50,239,683 are trainable.

**Loss**

```
total = ( MSE + (1 - SSIM) + L1 + w * KL(teacher || student) ) / PSNR
```

with `w = 0.4` and temperature `T = 0.8`. The teacher is fed `y_true` (the
clean patch), and both outputs go through a temperature-scaled softmax over the
channel axis before the KL term.

## Configuration (section 2)

| Name | Default | Meaning |
|---|---|---|
| `IMAGE_SIZE` | (256, 256) | Resize target |
| `PATCH_SIZE` | 64 | Patch side length |
| `CHANNELS` | 3 | Images are read as BGR by OpenCV |
| `NOISE_STDDEV` | 0.01 | Speckle sigma |
| `VAL_SPLIT` | 0.3 | Validation share of the train folder |
| `AUG_BATCH_SIZE` | 64 | Batch size of the augmentation generators |
| `MAX_TRAIN_BATCHES` / `MAX_VAL_BATCHES` / `MAX_TEST_BATCHES` | 40 / 10 / 2 | Batches of 64 written to disk |
| `TRAIN_BATCH_SIZE` | 32 | Batch size of `model.fit` **and** of the evaluation loop |
| `EPOCHS` | 100 | Training epochs |
| `LEARNING_RATE` | 5e-4 | Student Adam learning rate (beta_1 = 0.7) |
| `DISTILLATION_WEIGHT` / `DISTILLATION_TEMPERATURE` | 0.4 / 0.8 | Distillation term |
| `N_CONV_BLOCKS` / `STUDENT_FILTERS` / `STUDENT_DENSE_UNITS` | 16 / 64 / 512 | Student depth and width |
| `TEACHER_LEARNING_RATE` | 1e-3 | Teacher Adam learning rate (compiled but never used) |
| `LR_REDUCE_FACTOR` / `LR_REDUCE_PATIENCE` / `LR_MIN` | 0.8 / 3 / 1e-10 | `ReduceLROnPlateau` on `val_peak_signal_noise_ratio` |
| `DATA_TRAIN_DIR`, `DATA_TEST_DIR`, `SAMPLE_IMAGE_PATH` | Kaggle `../input/...` | Source dataset |
| `TRAIN_CLEAN_DIR` ... `TEST_NOISY_DIR` | `/kaggle/working/...` | Where the generated pairs are stored |

Note that there are **two different batch sizes**, and this is deliberate: the
augmentation generators write images in batches of 64, while training and
evaluation run in batches of 32. In the single-cell original this was one
variable (`batch_size`) being re-assigned halfway down the notebook, which made
the two values easy to confuse.

## Requirements

- Python 3.10
- TensorFlow / Keras 3 (uses `LeakyReLU(negative_slope=...)`; on Keras 2 change it back to `alpha=...`)
- numpy, pandas, matplotlib, scipy, scikit-image, opencv-python, Pillow
- A GPU is strongly recommended. The original run took about 3.8 hours for 100 epochs.

```bash
pip install tensorflow numpy pandas matplotlib scipy scikit-image opencv-python pillow
```

## Running

1. Place the dataset so that `DATA_TRAIN_DIR` and `DATA_TEST_DIR` point to folders with one sub-folder per class (required by `flow_from_directory`).
2. If you are not on Kaggle, change `WORK_DIR` and the dataset paths in section 2.
3. Run all cells top to bottom. Section 4 only needs to run once; later runs can skip it as long as the generated PNG folders still exist.

## Results from the original run

Test set: 2,048 patches from 128 images, speckle sigma = 0.01.

| PSNR (dB) | SSIM | MSE | RMSE | CNR | ENL | EPI | FOM | NIQE |
|---|---|---|---|---|---|---|---|---|
| 47.77 | 0.9955 | 2.7e-5 | 0.0045 | 80.29 | 9.11 | 0.0036 | 0.9964 | 0.206 |

`model.evaluate` also reported a test loss of 1.33e-4, an MSE of 2.69e-5 and a
test "accuracy" of 23.2 % (see caveat 5 below). Training used 36,240 noisy/clean
patches, with 8,256 for validation.

These outputs from the original run are still stored in the notebook, attached
to the cells that produced them.

## What changed when the notebook was split into sections

The code was reorganised and commented. Behaviour is intended to be identical,
apart from the following:

- The three copy-pasted save loops became `save_clean_noisy_pairs`; the
  repeated conv stacks in the teacher became `conv_block`; and the student's 16
  blocks are now generated by a loop driven by `N_CONV_BLOCKS`.
- `calculate_psnr_ssim` was renamed `calculate_metrics`, and its nine metrics
  are returned in the same order as before.
- The two `add_speckle_noise` functions were renamed `add_speckle_noise_pil` and
  `add_speckle_noise_array`. The original defined `add_speckle_noise` twice, the
  second definition silently overwriting the first.
- The sample-image reconstruction now uses `reconstruct_image` instead of a
  hard-coded 4x4 row/column counter.
- `LeakyReLU(alpha=...)` became `negative_slope=...`; the old name raises a
  deprecation warning in Keras 3.
- Every constant now has an upper-case name and a comment, and all paths live in
  section 2. The original re-assigned `image_size`, `batch_size` and
  `noise_stddev` in the middle of the file.
- Removed as dead code: `extract_patch` (never called), `swish_activation` and
  `swish_layer` (never used), `lr_schedule` and `early_stopping` (created but
  never passed to `fit`), a second full import block pointing at a different
  dataset (Berkeley / BSD400), and the unused `data_path_noisy`,
  `data_path_clean` and second `data_test_path` variables.
- In the ENL calculation the `std == 0` guard was overwritten by the next line
  in the original, so it was dropped. A zero std would still give a division
  warning.
- Saved outputs from the original run were re-attached to the cells that
  produced them.

## Things worth checking

These are in the code and were deliberately left unchanged:

1. **The teacher is never trained.** It is compiled but `fit` is never called on it, so the distillation term uses a randomly initialised network. If you want real distillation, train the teacher first (noisy -> clean).
2. **Teacher output is 62x62**, because the last conv uses the default `padding='valid'`. It is resized to 64x64 inside the loss.
3. **Validation data is augmented**, because train and validation share one generator. Validation images are also random augmentations of images from the same pool as training, so the validation score is optimistic.
4. **Train/validation overlap in practice.** Generators repeat, so 453 training images become 2,265 saved images (about 5 augmentations each).
5. **The `accuracy` metric is not meaningful** for image regression. The test "accuracy" of 23 % in the original run says nothing about quality, so use PSNR and SSIM instead.
6. **Only the first 128 of 133 test images** are used (`MAX_TEST_BATCHES = 2` with batch size 64).
7. **Evaluation drops the last partial batch** (`len // batch_size`).
8. **Metric quirks.** `compute_niqe` is a simplified stand-in, not the real NIQE. A zero standard deviation in ENL would still cause a division warning.
9. **Dividing the loss by PSNR** can behave oddly: if MSE reaches exactly 0, PSNR becomes 0 and the loss is infinite.
10. **The test set is noised once** with random noise and no seed, so results vary slightly between runs. Consider `np.random.seed(...)` for reproducibility.

## Files

| File | What it is |
|---|---|
| `mk-model.ipynb` | The notebook, split into 11 documented sections |
| `mk-model.original-backup.ipynb` | The original single-cell notebook, unchanged |
| `README.md` | This file |
