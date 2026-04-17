# Knowledge Graph-Based Semantic Scene Understanding for Autonomous Driving in Dilemma Situations

Official code for our paper accepted at **KSAE 2026 Spring Conference** (한국자동차공학회 춘계학술대회).

**Authors:** Woongje Cho, Hyeonseo Oh, Junseok Lee, Shiho Kim*
**Affiliation:** Yonsei University

## Abstract

For autonomous vehicles to operate safely in real-world environments, they must go beyond object detection and understand semantic relations and situational context within a scene. In particular, dilemma situations involving conflicting constraints such as accident avoidance, pedestrian priority, and traffic rule compliance are difficult to resolve using conventional 3D Scene Graphs (3DSGs), which mainly represent spatial structure. This paper proposes a Knowledge Graph (KG)-enhanced semantic scene understanding framework tailored to autonomous driving dilemma scenarios.

**Key results:**
- KG significantly outperforms 3DSG baseline (mean **4.41 vs. 3.56**, Wilcoxon p<0.001, Cohen's d=0.84)
- Largest gains in dilemma reasoning (+1.71)
- 5-condition ablation: LLM-only (1.25) -> KG-structure-only (3.29) -> 3DSG (3.56) -> 3DSG+NL (4.14) -> **KG full (4.41)**

## Repository Structure

```
├── experiment/
│   ├── experiment_v2.py           # Main KG vs 3DSG experiment (N=300 evaluations)
│   ├── experiment_ablation_v2.py  # 5-condition ablation study
│   ├── ground_truth.json          # 30 scene-understanding queries (6 cognitive categories)
│   └── scene_graph.json           # Enhanced 3DSG data (48 nodes, 23 edges)
├── ontology/
│   ├── scene_tools_kg.py          # KG retrieval via SPARQL (GraphDB)
│   ├── scene_tools_dsg.py         # 3DSG graph traversal retrieval
│   └── scene_tools.py             # Shared utilities
├── results/
│   ├── experiment_results_v2.json     # Full experiment results
│   └── experiment_results_ablation_v2.json  # Ablation results
└── README.md
```

## Setup

### Requirements
- Python 3.9+
- GraphDB (for KG/SPARQL queries) -- download from [Ontotext](https://www.ontotext.com/products/graphdb/)
- OpenAI API key

```bash
pip install -r requirements.txt
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-key-here"
```

GraphDB must be running locally at `http://localhost:7200` with the `DrivingKG` repository loaded (see ontology/ for OWL files).

## Running Experiments

### Main experiment (KG vs 3DSG)
```bash
python experiment/experiment_v2.py
```

### Ablation study
```bash
python experiment/experiment_ablation_v2.py
```

## Citation

```bibtex
@inproceedings{cho2026kg,
  title={Knowledge Graph-Based Semantic Scene Understanding for Autonomous Driving in Dilemma Situations},
  author={Cho, Woongje and Oh, Hyeonseo and Lee, Junseok and Kim, Shiho},
  booktitle={Proceedings of the KSAE Spring Conference},
  year={2026}
}
```

## License

MIT License
