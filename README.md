---
title: visual-attention-intervention
app_file: app.py
sdk: gradio
sdk_version: 6.5.1
---
# Survey of Visual Attention Intervention Methods for Spatial Reasoning

Implementation of three visual attention intervention methods for Qwen3-VL to investigate their enhanced capabilities for visual spatial reasoning. Methods used are from the following papers:

- AdaptVis: [Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas](https://arxiv.org/pdf/2503.01773).
- VEA: [Seeing but not believing: Probing the disconnect between visual attention and answer correctness in vlms](https://arxiv.org/pdf/2510.17771).
- CLVS: [Cross-Layer Vision Smoothing: Enhancing Visual Understanding via Sustained Focus on Key Objects in Large Vision-Language Models](https://arxiv.org/pdf/2503.01773).

All experiments are done using the [Visual Spatial Reasoning](https://arxiv.org/pdf/2205.00363) dataset and the Qwen3-VL 2B model. Evaluations are done by scoring on the random and zeroshot dev splits from VSR. Further analysis can be done using the heatmap visualizations and the plots of text/image attention weights across layers, which are generated during the evaluation experiments.

## Results

Below is the accuracy scores on the random and zeroshot dev splits of the VSR dataset. We compare scores from Qwen3-VL 2B fine-tuned on the VSR random and zeroshot train split with the scores of the same fine-tuned model + the various visual attention intervention methods.

|                        | random (dev) | zeroshot (dev) |
| ---------------------- | ------------ | -------------- |
| Qwen3-VL 2B            | 0.8277       | 0.7588         |
| Qwen3-VL 2B + AdaptVis | **0.8332**   | 0.7676         |
| Qwen3-VL 2B + VEA      | 0.8304       | 0.7441         |
| Qwen3-VL 2B + CLVS     | 0.8268       | **0.7735**     |

## Setup

### Download VSR

Download the VSR data by cloning the [visual-spatial-reasoning](https://github.com/cambridgeltl/visual-spatial-reasoning) repository and following the steps in `data/README.md`

### Download COCO Annotations

Download the [COCO 2017 Train/Val annotations](https://cocodataset.org/#download). The VSR images are taken from COCO, and these annotations are used for the VEA method.

### Install Project

Install [Poetry](https://python-poetry.org/)

Then install dependencies and activate the poetry environment
```
poetry install
poetry shell
```

## Fine-Tuning

Finetune Qwen3-VL on the VSR random or zeroshot dataset splits. I've found the best results with freezing the image encoder. You may also finetune with LoRA by using `--use_lora` flag and setting the `--lora_r` and `--lora_alpha` parameters.

Optionally visualize training on [Aim](https://github.com/aimhubio/aim) with the `--log_with_aim` flag.

```
python scripts/finetuning_qwen3vl.py
    --dataset_type "random" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --freeze_vision \
    --log_with_aim
```

## Evaluation

Evaluate on the VSR random or zeroshot dataset using the dev or test split. 

### Standard Evaluation

Standard evaluation of Qwen3-VL with no visual attention intervention.

```
python scripts/evaluating_qwen3vl.py 
    --dataset_type "random" \
    --dataset_split "dev" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --model_id "./path/to/model/checkpoint" \
```

### AdaptVis Evaluation

Evaluation of Qwen3-VL using AdaptVis. The default hyperparameters for random and zeroshot splits are shown below.

```
python scripts/evaluating_qwen3vl_adapt_vis.py 
    --dataset_type "random" \
    --dataset_split "dev" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --model_id "./path/to/model/checkpoint"
    --confidence_threshold 0.8 \
    --sharpen_weight 1.2 \
    --smoothen_weight 0.2
```

```
python scripts/evaluating_qwen3vl_adapt_vis.py 
    --dataset_type "zeroshot" \
    --dataset_split "dev" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --model_id "./path/to/model/checkpoint"
    --confidence_threshold 0.8 \
    --sharpen_weight 1.2 \
    --smoothen_weight 0.1
```

### VEA Evaluation

Evaluation of Qwen3-VL using VEA. The default hyperparameters for random and zeroshot splits are shown below.

```
python scripts/evaluating_qwen3vl_vea.py 
    --dataset_type "random" \
    --dataset_split "dev" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --model_id "./path/to/model/checkpoint" \
    --coco_annotations_dir "./path/to/coco/annotations" \
    --smooth_strength 0.5 \
    --highlight_strength 0.5
```

```
python scripts/evaluating_qwen3vl_vea.py 
    --dataset_type "zeroshot" \
    --dataset_split "dev" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --model_id "./path/to/model/checkpoint" \
    --coco_annotations_dir "./path/to/coco/annotations" \
    --smooth_strength 0.8 \
    --highlight_strength 0.2
```

### CLVS Evaluation

Evaluation of Qwen3-VL using CLVS. The default hyperparameters for random and zeroshot splits are shown below.

```
python scripts/evaluating_qwen3vl_clvs.py 
    --dataset_type "random" \
    --dataset_split "dev" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --model_id "./path/to/model/checkpoint" \
    --smoothing 0.7 \
    --window_memory_size 0.8 \
    --uncertainty_threshold 0.5
```

```
python scripts/evaluating_qwen3vl_clvs.py 
    --dataset_type "zeroshot" \
    --dataset_split "dev" \
    --vsr_data_dir "./path/to/visual-spatial-reasoning/data" \
    --model_id "./path/to/model/checkpoint" \
    --smoothing 0.7 \
    --window_memory_size 0.8 \
    --uncertainty_threshold 0.5
```

## Citations

```bibtex
@article{Liu2022VisualSR,
    title={Visual Spatial Reasoning},
    author={Fangyu Liu and Guy Edward Toh Emerson and Nigel Collier},
    journal={Transactions of the Association for Computational Linguistics},
    year={2023},
}

@misc{chen2025spatialreasoninghardvlms,
    title={Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas}, 
    author={Shiqi Chen, Tongyao Zhu, Ruochen Zhou, Jinghan Zhang, Siyang Gao, Juan Carlos Niebles, Mor Geva, Junxian He, Jiajun Wu, Manling Li},
    year={2025},
    eprint={2503.01773},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2503.01773}, 
}

@article{
    title={Seeing but not believing: Probing the disconnect between visual attention and answer correctness in vlms}, 
    author={Zhining Liu, Ziyi Chen, Hui Liu, Chen Luo, Xianfeng Tang, Suhang Wang, Joy Zeng, Zhenwei Dai, Zhan Shi, Tianxin Wei, Benoit Dumoulin, Hanghang Tong},
    year={2025},
    eprint={2510.1777},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/pdf/2510.17771}, 
}

@article{
    title={Cross-Layer Vision Smoothing: Enhancing Visual Understanding via Sustained Focus on Key Objects in Large Vision-Language Models}, 
    author={Jianfei Zhao, Feng Zhang, Xin Sun, Chong Feng, Zhixing Tan},
    year={2025},
    eprint={2509.12897},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/pdf/2503.01773}, 
}
```
