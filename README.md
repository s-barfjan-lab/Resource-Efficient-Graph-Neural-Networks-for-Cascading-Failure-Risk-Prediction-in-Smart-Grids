# Resource-Efficient Graph Neural Networks for Cascading Failure Risk Prediction in Smart Grids

This project develops and compares graph-based deep learning models for predicting cascading failure risk in smart grids.
The goal is to build an intelligent monitoring pipeline able to classify whether a simulated power-grid scenario leads to a cascade and to estimate the severity of the event through the DNS value, Demand Not Served.

The project was developed for the course **Intelligent Monitoring and Control Systems**.

---

## Project Overview

Cascading failures are critical events in power systems, where the failure of one or more transmission lines can propagate through the network and lead to wider instability. Since power grids are naturally graph-structured systems, this project investigates whether Graph Neural Networks can improve risk prediction compared with a non-graph baseline.

The project compares:

* **FCNN baseline**
  A non-graph neural network that uses aggregated node and edge features without explicitly modelling the grid topology.

* **Edge-aware GINE baseline**
  A non-attention graph neural network using topology, node features, and edge features.
  This model is used to test whether edge-aware message passing alone is sufficient for cascading failure prediction.

* **Manual GAT**
  A manually designed Graph Attention Network using graph topology, node features, edge features, and attention mechanisms.

* **NAS-GAT**
  A Graph Attention Network selected through a constrained Neural Architecture Search procedure.

* **Head-pruned NAS-GAT**
  A compressed version of the NAS-GAT obtained through structured attention-head pruning and fine-tuning.

---

## Dataset

The project uses the **PowerGraph cascading failure dataset**, focusing on the graph-level cascading failure prediction task.

The notebook automatically downloads and extracts the PowerGraph archive, which contains data for:

* IEEE-24
* IEEE-39
* IEEE-118
* UK grid

The final experiments were run on:

```python
GRID_NAME = "ieee24"
FAST_RUN = False
```

For the IEEE-24 grid, the processed dataset contains:

| Item                            |           Value |
| ------------------------------- | --------------: |
| Simulated scenarios             |          21,500 |
| Buses / nodes                   |              24 |
| Physical branches               |              38 |
| Bidirectional graph edges       |              76 |
| Node features per bus           |               3 |
| Edge features per branch        |               5 |
| Train / validation / test split | 70% / 15% / 15% |

---

## Main Preprocessing Choices

Each cascading-failure scenario is converted into one PyTorch Geometric graph.

Each graph contains:

* `x`: node feature matrix
* `edge_index`: graph topology
* `edge_attr`: edge feature matrix
* `y_class`: binary label, cascade or no cascade
* `y_reg`: regression label for DNS severity
* `edge_mask`: explanation mask from the dataset
* `idx`: original sample index

A key correction was applied during preprocessing:

> Zero-valued edge rows are not removed.
> Instead, they are preserved and an additional binary feature, `initial_failed_flag`, is added.

This is important because all-zero edge features may contain information about initially failed or unavailable lines. Removing them would cause information loss and would also alter the original grid topology.

The final edge feature vector therefore contains:

```text
4 original electrical edge features + 1 initial_failed_flag
```

---

## Model Architecture

All neural models are trained in a multitask setting with two output heads:

1. **Classification head**
   Predicts whether the scenario leads to a cascade.

2. **Regression head**
   Predicts DNS severity.

The multitask loss is:

```text
loss = BCEWithLogitsLoss(classification) + beta * MSELoss(log1p(DNS))
```

where:

```python
BETA_REG = 0.3
```

The DNS target is transformed using:

```python
log1p(DNS)
```

This reduces the skewness of the regression target and makes training more stable.

---

## Training Pipeline

The full pipeline includes:

1. Installing required packages
2. Downloading and extracting the PowerGraph dataset
3. Selecting the target grid
4. Loading `.mat` files
5. Converting each sample into a PyTorch Geometric graph
6. Performing exploratory data analysis
7. Splitting the dataset into train, validation, and test sets
8. Normalizing node and continuous edge features using only the training set
9. Training baseline models
10. Running Neural Architecture Search for GAT configurations
11. Applying structured attention-head pruning
12. Fine-tuning the pruned model
13. Selecting the best classification threshold on the validation set
14. Evaluating final models on the test set
15. Producing plots, confusion matrices, regression metrics, and attention-based interpretability analysis

---

## Neural Architecture Search

The NAS procedure performs a constrained random search over GAT configurations.

The search space includes:

```python
search_space = {
    "hidden_dim": [32, 64],
    "num_layers": [2, 3],
    "heads": [2, 4],
    "dropout": [0.1, 0.2, 0.3],
    "pooling": ["mean", "max"],
    "mlp_hidden": [32, 64],
    "lr": [5e-4, 1e-3],
}
```

The NAS objective rewards predictive performance while penalizing model size and latency:

```text
score = F1 + 0.2 * AUROC - 1e-6 * parameters - 0.002 * latency
```

This encourages the search to find models that are accurate but also reasonably efficient.

---

## Structured Attention-Head Pruning

After selecting the best NAS-GAT model, structured pruning is applied to reduce the number of attention heads.

The pruning method estimates the importance of each attention head using the magnitude of its learned parameters.
The least important heads are removed, and the smaller model is fine-tuned.

In the reported run, the NAS-GAT configuration was pruned from:

```text
4 attention heads per layer
```

to:

```text
2 attention heads per layer
```

This reduced the parameter count from:

```text
13,986 parameters
```

to:

```text
8,162 parameters
```

---

## Threshold Tuning

Instead of using a fixed classification threshold of `0.5`, the project tunes the threshold on the validation set.

Thresholds from `0.05` to `0.95` are tested, and the threshold that maximizes validation F1-score is selected.

This is important because, in a monitoring system, false negatives are especially dangerous.
A false negative means that the model failed to detect a risky cascading failure scenario.

---

## Example Results

The exact values may vary slightly depending on hardware, random initialization, and runtime conditions.
The final notebook generates a complete comparison table including accuracy, precision, recall, F1-score, AUROC, DNS regression metrics, parameter count, model size, latency, and loss.

A representative summary from the final run is:

| Model               | Main Observation                                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| FCNN                | Useful non-graph baseline, but does not explicitly use grid topology.                                                       |
| Edge-aware GINE     | Best overall classification performance; edge-aware message passing was highly effective.                                   |
| Manual GAT          | Uses attention mechanisms but did not outperform the edge-aware GINE baseline.                                              |
| NAS-GAT             | Architecture search improved the attention-based model configuration, but the model remained slower than the GINE baseline. |
| Head-pruned NAS-GAT | Reduced model size substantially while preserving competitive performance after fine-tuning.                                |

One of the strongest reported models was the edge-aware GINE baseline, which achieved high classification performance while remaining more efficient than the attention-based GAT models.

---

## Interpretability

The notebook also includes an attention-based interpretability section.

For a selected cascade test case, the Manual GAT model returns attention weights for the edges.
The analysis compares highly attended edges with the ground-truth explanation mask from the dataset.

This helps inspect whether the model focuses on edges that are related to the actual cascading-failure explanation.

However, attention weights should be interpreted carefully.
They provide a useful diagnostic signal but should not be treated as a complete causal explanation.

---

## Requirements

The project uses Python with PyTorch and PyTorch Geometric.

Main dependencies:

```text
torch
torch-geometric
mat73
numpy
pandas
scikit-learn
matplotlib
requests
```

In Google Colab, the main installation command used in the notebook is:

```python
!pip install -q torch-geometric mat73 scikit-learn pandas matplotlib
```

---

## How to Run

### Option 1: Run in Google Colab

1. Open the notebook in Google Colab.
2. Run the installation cell.
3. Keep:

```python
GRID_NAME = "ieee24"
FAST_RUN = True
```

for a quick test.

4. For the final experiment, use:

```python
GRID_NAME = "ieee24"
FAST_RUN = False
```

5. Run all cells from top to bottom.

---

### Option 2: Run Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Install dependencies:

```bash
pip install torch torch-geometric mat73 numpy pandas scikit-learn matplotlib requests
```

Then run the notebook:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

## Suggested Repository Structure

```text
.
├── README.md
├── requirements.txt
└── notebooks/
    └── cascading_failure_gnn_powergraph.ipynb
```

---

## Reproducibility

The notebook uses a fixed random seed:

```python
SEED = 42
```

The dataset split is stratified by the binary cascade label:

```text
70% training
15% validation
15% test
```

Feature normalization is computed only on the training set to avoid data leakage.

---

## Key Takeaways

The main conclusion of the project is that edge-aware message passing is highly effective for cascading failure risk prediction.

Although the attention-based GAT models are more complex and offer interpretability through attention weights, the edge-aware GINE baseline achieved the strongest performance-efficiency trade-off in the reported IEEE-24 experiment.

The NAS and pruning stages remain useful because they show how attention-based GNNs can be optimized and compressed, but the results suggest that model complexity alone does not guarantee better performance.

---

## Future Work

Possible extensions include:

* Testing the same pipeline on IEEE-39, IEEE-118, and UK grids
* Evaluating cross-grid generalization
* Improving the NAS objective with stronger latency-aware constraints
* Exploring more advanced pruning strategies
* Adding calibration metrics for risk probabilities
* Comparing attention explanations with dedicated GNN explainability methods
* Deploying the best model in a lightweight monitoring prototype

---

## Project Status

This project is implemented as an experimental research notebook for intelligent monitoring of power grids using graph neural networks.

The current version includes:

* Full preprocessing pipeline
* Multitask classification and regression
* FCNN baseline
* Edge-aware GNN baseline
* Manual GAT
* NAS-GAT
* Structured head pruning
* Threshold tuning
* Final evaluation
* Confusion matrix analysis
* DNS regression analysis
* Attention-based interpretability
