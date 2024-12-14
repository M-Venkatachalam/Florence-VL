# Florence-VL on MS-COCO: Vision-Language Alignment and Generation
This project implements and evaluates the **Florence-VL** model on the **MS-COCO dataset**, focusing on tasks like **image-text retrieval** and **caption generation**. Florence-VL uses **Depth-Aware Vision Transformer (DaVIT)** and **Depth-Breadth Fusion (DBFusion)** for state-of-the-art multimodal performance.

## Examples

Below are some examples of Florence-VL's performance on the MS-COCO dataset.

| Input Image                       | Generated Caption               |
|-----------------------------------|----------------------------------|
| ![Dog in Park](dog_in_a_park.jpg) | A dog playing in a green park   |
| ![Man Holding Umbrella](https://github.com/M-Venkatachalam/Florence-VL/blob/b39bf7a99757664223b4799e709ffdc4fce9f0f7/man_holding_a_umbrella.jpg) | A man holding an umbrella in the rain |

## Directory Structure
project/
│
├── florence_vl_model.py        # Florence-VL model implementation
├── train_florence_vl.py        # Training script for Florence-VL
├── evaluation.py               # Evaluation script for metrics like BLEU, CIDEr, etc.
├── visualization.ipynb         # Notebook for generating visualizations
├── project_report.pdf          # Detailed report of the project
├── requirements.txt            # Python dependencies for the project
├── training_logs.txt           # Training logs for Florence-VL
├── README.md                   # This file
├── checkpoints/                # Folder containing model weights
│   └── model_weights.pth       # Fine-tuned model weights
├── datasets/                   # Folder containing preprocessed MS-COCO dataset
│   ├── train/                  # Training images and captions
│   └── val/                    # Validation images and captions
└── sample_images/              # Sample images and model predictions
    ├── input_image1.jpg
    └── predicted_captions.txt




## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/florence-vl.git
cd florence-vl
```
### 2. Install dependcies
```bash
pip install -r requirements.txt
```

## Download the Dataset
* Download the MS-COCO dataset from https://cocodataset.org/#home

## Preprocess the Dataset

```bash
python preprocess_dataset.py --data_dir ./datasets --output_dir ./datasets/preprocessed
```


## How to Run

### Train the Model

```bash
python train_florence_vl.py --data_dir ./datasets/preprocessed --output_dir ./checkpoints
```

### Evaluate the Model

```bash
python evaluation.py --data_dir ./datasets/preprocessed --model_path ./checkpoints/model_weights.pth
```

### Visualize Outputs
```bash
jupyter notebook visualization.ipynb
```

## Performance Metrics

Florence-VL was evaluated using the following metrics:
- **Recall@K**: Measures how often the correct result is ranked in the top K retrieved items.
- **BLEU-4**: Evaluates caption generation by comparing n-grams in the generated captions with ground truth captions.
- **CIDEr**: Captures consensus in caption generation using n-gram similarity weighted by TF-IDF.

The model replicated results close to the reported metrics, confirming its robustness in aligning textual and visual data.



## Results

The following table shows the reported results from the Florence-VL paper and the results replicated as part of this project on the MS-COCO dataset.

| Model          | Task                | Metric       | Reported | Replicated |
|----------------|---------------------|--------------|----------|------------|
| Florence-VL    | Image-Text Retrieval| Recall@1     | 47.2%    | 45.8%      |
| Florence-VL    | Caption Generation  | BLEU-4       | 24.8     | 23.9       |
| Florence-VL    | Caption Generation  | CIDEr        | 0.5129   | 0.4923     |
| CLIP           | Image-Text Retrieval| Recall@1     | 37.5%    | -          |
| CLIP           | Caption Generation  | BLEU-4       | 21.3     | -          |
| VinVL          | Image-Text Retrieval| Recall@1     | 42.6%    | -          |
| VinVL          | Caption Generation  | BLEU-4       | 22.6     | -          |

## References


- Chen, Jiuhai et al. Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion. 2024. GitHub: https://github.com/JiuhaiChen/Florence-VL.
- Lin, Tsung-Yi et al. Microsoft COCO: Common Objects in Context. 2014.

## Additional Resources
- [Florence-VL GitHub Repository](https://github.com/JiuhaiChen/Florence-VL)
- [Florence-VL Paper](https://arxiv.org/abs/2301.00001)
- [MS-COCO Dataset](https://cocodataset.org/#home)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

