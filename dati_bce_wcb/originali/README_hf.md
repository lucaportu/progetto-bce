---
dataset_info:
- config_name: '5768'
  features:
  - name: sentences
    dtype: string
  - name: stance_label
    dtype: string
  - name: time_label
    dtype: string
  - name: certain_label
    dtype: string
  - name: year
    dtype: int64
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 168266
    num_examples: 700
  - name: test
    num_bytes: 36991
    num_examples: 150
  - name: val
    num_bytes: 37606
    num_examples: 150
  download_size: 114274
  dataset_size: 242863
- config_name: '78516'
  features:
  - name: sentences
    dtype: string
  - name: stance_label
    dtype: string
  - name: time_label
    dtype: string
  - name: certain_label
    dtype: string
  - name: year
    dtype: int64
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 171121
    num_examples: 700
  - name: test
    num_bytes: 34464
    num_examples: 150
  - name: val
    num_bytes: 37278
    num_examples: 150
  download_size: 114154
  dataset_size: 242863
- config_name: '944601'
  features:
  - name: sentences
    dtype: string
  - name: stance_label
    dtype: string
  - name: time_label
    dtype: string
  - name: certain_label
    dtype: string
  - name: year
    dtype: int64
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 168394
    num_examples: 700
  - name: test
    num_bytes: 37747
    num_examples: 150
  - name: val
    num_bytes: 36722
    num_examples: 150
  download_size: 114882
  dataset_size: 242863
configs:
- config_name: '5768'
  data_files:
  - split: train
    path: 5768/train-*
  - split: test
    path: 5768/test-*
  - split: val
    path: 5768/val-*
- config_name: '78516'
  data_files:
  - split: train
    path: 78516/train-*
  - split: test
    path: 78516/test-*
  - split: val
    path: 78516/val-*
- config_name: '944601'
  data_files:
  - split: train
    path: 944601/train-*
  - split: test
    path: 944601/test-*
  - split: val
    path: 944601/val-*
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
language:
- en
tags:
- finance
- econ
pretty_name: european_central_bank
size_categories:
- 1K<n<10K
---

## Dataset Summary
For dataset summary, please refer to [https://huggingface.co/datasets/gtfintechlab/european_central_bank](https://huggingface.co/datasets/gtfintechlab/european_central_bank)

## Additional Information
This dataset is annotated across three different tasks: Stance Detection, Temporal Classification, and Uncertainty Estimation. The tasks have four, two, and two unique labels, respectively. This dataset contains 1,000 sentences taken from the meeting minutes of the European Central Bank. 

### Label Interpretation

#### Stance Detection
- **Hawkish:** The sentence supports contractionary monetary policy.
- **Dovish:** The sentence supports expansionary monetary policy.
- **Neutral:** The sentence contains neither hawkish or dovish sentiment, or both hawkish and dovish sentiment.
- **Irrelevant:** The sentence is not related to monetary policy.

#### Temporal Classification
- **Forward-looking:** The sentence discusses future economic events or decisions.
- **Not Forward-looking:** The sentence discusses past or current economic events or decisions.

#### Uncertainty Estimation
- **Certain:** Indicates that the sentence presents information definitively.
- **Uncertain:** Indicates that the sentence presents information with speculation, possibility, or doubt.

## Licensing Information
The european_central_bank dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. [More information in the paper.](https://arxiv.org/)

## Citation Information
```bibtex
@article{WCBShahSukhaniPardawala,
  title={Words That Unite The World: A Unified Framework for Deciphering Global Central Bank Communications},
  author={Agam Shah, Siddhant Sukhani, Huzaifa Pardawala et al.},
  year={2025}
}
```
## Contact
For any european_central_bank dataset related issues and questions, please contact:

- Huzaifa Pardawala: huzaifahp7[at]gatech[dot]edu

- Siddhant Sukhani: ssukhani3[at]gatech[dot]edu

- Agam Shah: ashah482[at]gatech[dot]edu

## GitHub Link
[Link to our GitHub repository.](https://github.com/gtfintechlab/WorldsCentralBanks) 
