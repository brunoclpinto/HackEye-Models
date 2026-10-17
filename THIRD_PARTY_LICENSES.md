# Third-party licences and acknowledgements

HackEye-Models trains the segmentation models of [HackEye](https://github.com/brunoclpinto/HackEye), a
free, open-source, **non-commercial** project. This repository's own code (training and conversion
scripts) is under the [Apache License 2.0](LICENSE).

**The trained models in `models/` are not Apache-2.0.** They inherit the terms below. That is what this
file is about.

> This is a good-faith summary, not legal advice. The linked licence texts are what bind.

Last reviewed: 2026-10-17. The app and RnD side is covered in HackEye's
[THIRD_PARTY_LICENSES.md](https://github.com/brunoclpinto/HackEye/blob/main/THIRD_PARTY_LICENSES.md).

---

## 1. Requires legal handling

| Component | Used for | Licence / terms | What it means |
|---|---|---|---|
| **Ultralytics** (library and YOLO26 weights: `yolo26*-sem-ade20k.pt`, `yolo26s-sem.pt`) | The whole training rig, and the warm-start checkpoints of every model in `models/` | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | Ultralytics treats models trained or fine-tuned with its library as covered by AGPL-3.0 unless an Enterprise licence is bought. **`models/*HackEye*.pt` are therefore distributed under AGPL-3.0**: whoever receives or serves them must be able to get the complete corresponding source, which is this repository at the matching commit. |
| **Mapillary Vistas v2.0** | Training data of every `*HackEye*` model | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) plus Mapillary's research licence ([mapillary.com/dataset/vistas](https://www.mapillary.com/dataset/vistas)); see `training/datasets/vistas-raw/LICENSE` | **Don't redistribute the data.** It is gitignored; each user registers and downloads it themselves. The models are for **non-commercial use, with attribution to Mapillary**. Whether ShareAlike extends to trained weights is unsettled; this project takes the cautious reading and releases them only under non-commercial, attribution-preserving terms that are compatible with it. |
| **ADE20K** | The data behind Ultralytics' `yolo26*-sem-ade20k.pt` warm-start | [ADE20K terms](https://groups.csail.mit.edu/vision/datasets/ADE20K/): research and non-commercial use | Non-commercial use, with citation (Zhou et al., CVPR 2017). |
| **Cityscapes** | The data behind Ultralytics' `yolo26s-sem.pt`. It was tried as a warm-start in the A/B grid and rejected, so it isn't in the released models | [Cityscapes terms](https://www.cityscapes-dataset.com/license/) | Only relevant if a future model starts from that checkpoint. |

**Released models, in short:** `models/*HackEye*.pt` are free for non-commercial use under AGPL-3.0, with
attribution to Ultralytics, Mapillary Vistas and ADE20K.

---

## 2. Thanks

These permissively licensed projects only ask for attribution:
- **PyTorch**: BSD-3-Clause.
- **OpenCV** (`opencv-python-headless`): Apache-2.0.
- **NumPy**: BSD-3-Clause.
- **Pillow**: MIT-CMU.
- **PyYAML**: MIT.
- **polars**: MIT.
- The `pytorch/pytorch` base image.

Thank you to their authors and maintainers, and to Mapillary, MIT CSAIL (ADE20K) and Ultralytics for
making their data and models available to non-commercial projects like this one.
