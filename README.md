# Mechanics-informed graph neural networks reveal composition-porosity-strength relationships in bone biomaterials

MI-GNN predicts compressive strength, elastic modulus, and yield strength from mesoscale material-phase graphs. Edge-gated message passing represents phase connectivity, while differentiable Gibson–Ashby scaling, Hooke consistency, and a porosity-dependent yield envelope constrain the shared constitutive representation.

## Installation

Python 3.10 and CUDA 12.1 are supported by the pinned environment.

```bash
conda env create -f environment.yml
conda activate mignn
pip install --no-deps -e .
```

The container path uses PyTorch 2.2.2 with CUDA 12.1.

```bash
docker build -t mignn .
```

## Data

The study defines CerScaff-Lit with 847 records, MetScaff-TPMS with 1,240 records, and PolyComScaff with 632 records. The manuscript states that these curated artifacts will be deposited in Zenodo upon publication, but it gives no DOI or downloadable archive. `datasets.txt` records this status and intentionally contains no speculative link. Place licensed tabular exports under `datasets/raw` and retain source-publication identifiers in every row.

Expected fields include composition fractions, porosity, pore size, density, solid-phase density and strength, manufacturing route, architecture, compressive strength, elastic modulus, yield strength, and elastic-limit strain. Literature scaffolds are represented with 8–25 phase-region nodes. TPMS samples use a 5 × 5 × 5 grid with 26-neighbor solid-voxel connectivity.

## Preparation

```bash
mignn-prepare --input datasets/raw --output datasets/processed
```

Data are split by material system and porosity band into 70% training, 15% validation, and 15% test partitions. Composition OOD evaluation holds out HA:TCP ratios for CerScaff-Lit, I-WP architecture for MetScaff-TPMS, and the PLA matrix for PolyComScaff.

## Training

```bash
mignn-train --config configs/main.yaml
mignn-train --config configs/without_gibson_ashby.yaml
mignn-train --config configs/without_hooke.yaml
mignn-train --config configs/without_yield.yaml
mignn-train --config configs/without_all_physics.yaml
```

The main configuration uses four message-passing layers, 128 hidden features, batch size 32, AdamW at learning rate 0.001 and weight decay 0.00001, cosine warm restarts every 50 epochs, 50 physics-only epochs, and 200 joint epochs. Early stopping patience is 30. The five reported seeds are 42, 1024, 2048, 3407, and 7.

## Evaluation

```bash
mignn-evaluate --predictions artifacts/predictions.csv --targets datasets/processed/test.csv
```

Evaluation reports R², MAE, RMSE, and violation rates for Gibson–Ashby scaling, Hooke consistency, and the yield envelope. Reported compressive-strength R² values are 0.957 ± 0.006 for CerScaff-Lit, 0.979 ± 0.003 for MetScaff-TPMS, and 0.931 ± 0.007 for PolyComScaff across five seeds.

## Compute budget

The reported D1 run uses one NVIDIA A100 40 GB GPU and takes about 45 minutes. Reported inference latency is 4.8 ms per scaffold. Storage was not reported. The implementation does not inflate these requirements beyond the experiment described in the manuscript.

## Scientific mapping

Edge-gated node updates and learned edge updates correspond to equations 1 and 2. Attention readout corresponds to equation 3. The combined objective and its data, Gibson–Ashby, Hooke, and yield terms correspond to equations 4–9. The configuration set covers the physics-loss ablations in Table 2; the main configuration follows the selected architecture, batch size, and physics weights in Supplementary Table S4. The manuscript contains a learning-rate discrepancy: Methods reports 0.0005, while Supplementary Table S4 identifies 0.001 as the selected value. The main configuration uses the selected value from the table, and `provenance.json` records that resolution.

## License

The software is distributed under the MIT License. Dataset licenses remain governed by their source publications and future archive deposit.
