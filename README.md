# AdvEval
This repo contains the official code of ACL24 findings paper "Unveiling the Achilles' Heel of NLG Evaluators: A Unified Adversarial Framework Driven by Large Language Models".

### Setup
#### Install Environment
```shell
conda create -n adveval python=3.9
conda activate adveval
pip install -r requirments
```


#### Configure API Keys
```shell
export OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
export PALM_API_KEY=[YOUR_PALM_API_KEY]
```

### Evaluation
Example command to evaluate UniEval on dialogue response generation task:
```shell
# To obtain R+ samples
python main.py \
    --task_name response_dailydialog \
    --gold_evaluator_name palm_dialog \
    --victim_evaluator_name unieval_dialog \
    --optimizer_name optimizer_dialog_palm

# To obtain R- samples
python main.py \
    --task_name response_dailydialog \
    --gold_evaluator_name palm_dialog \
    --victim_evaluator_name unieval_dialog \
    --optimizer_name optimizer_dialog_palm \
    --negative_optimization_goal
```

### How to Cite

Please cite our paper:

```
@article{chen2024adveval,
  title={Unveiling the Achilles' Heel of NLG Evaluators: A Unified Adversarial Framework Driven by Large Language Models},
  author={Chen, Yiming and Zhang, Chen and Luo, Danqing and D'Haro, Luis Fernando and Tan, Robby T and Li, Haizhou},
  journal={arXiv preprint arXiv:2405.14646},
  year={2024}
}
```
