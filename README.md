# Modeling Protagonist Emotions for Emotion-Aware Storytelling

This repository contains **preliminary** code and data for the paper titled:

[Modeling Protagonist Emotions for Emotion-Aware Storytelling](https://arxiv.org/abs/2010.06822)

*Faeze Brahman, and Snigdha Chaturvedi.* EMNLP 2020.

I'm still cleaning up the code! Full code documentation will be ready soon! 

## Dataset: ROCStories
The dataset can be downloaded from [here](https://drive.google.com/file/d/17UhNDjAvkm2BlFNTWVWibO2ak1VywAc5/view?usp=sharing) and unzipped in `data/` folder.

**Data files includes**:
1. `[train/test/dev]_x1.txt`: titles
2. `[train/test/dev]_x4.txt`: stories
3. `[train/test/dev]_mapped.txt`: automatically annotated emotion arcs

## Code

* The code depends on [Texar](https://github.com/asyml/texar). Please install the version under [third_party/texar](./third_party/texar). Follow the installation instructions in the README there.
* Download gpt-2-M from [here](https://github.com/openai/gpt-2) and put it in `gpt2_pretrained_models/` folder.
* The BERT-based classifier is trained using [fast-bert](https://github.com/kaushaltrivedi/fast-bert). Please git clone (or `pip install`) it and use `run_classifier_bert.py` to train the emotion classifier.
* For obtaining emotional reactions, please git clone [COMET](https://github.com/atcbosselut/comet-commonsense) here. And move `comet_generate.py` and `find_x_o_appx.py` there. 
* Use `prepare_data.py` to preprocess the story data and transform them into TFRecord format. An example command is (please see the code for more config options).
```bash
python prepare_data.py --data_dir=data

```
* Run `run_[X].sh` for training/testing model `[X]`. (please see config files for more config options.)
* Use `Reinforcement/run_evaluation.py` for evaluation on emotion faithfulness. An example command is:
```bash
python Reinforcement/run_evaluation.py --all-preds-dir <PATH_TO_GENERATED_TSV_FILE> --arc-file <PATH_TO_ARC_FILE>  --output_file <PATH_TO_SAVE_JSON_RESULTS>
```
* BLEU scores measurements:
```bash
perl LIB/multi-bleu.perl data/test_x4.txt < <PATH_TO_GENERATED_TXT_FILE>
```
* The `Distinct-n` scores in the paper use the code [here](https://github.com/abisee/story-generation-eval).

The code is adapted from [Counterfactual Story Generation](https://github.com/qkaren/Counterfactual-StoryRW).


#### Reference

Please cite our paper using the following bibtex:
```
@inproceedings{emo-aware-2020,
  title={Modeling Protagonist Emotions for Emotion-Aware Storytelling},
  author={Faeze Brahman and Snigdha Chaturvedi},
  booktitle={EMNLP},
  year={2020}
}
```
