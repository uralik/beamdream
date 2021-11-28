# Importance of a Search Strategy in Neural Dialogue Modelling

[https://uralik.github.io/beamdream/](https://uralik.github.io/beamdream/)

This repo provides code, trained models, run scripts and human evaluation transcripts for our work
on different search strategies for neural dialogue models.

## Code

All the training and human evaluation was done using ParlAI framework. ParlAI is being actively
developed and we do not confirm our code to be working with master branch code. 

It should be working using ParlAI commit `5899c07934836d8757ebfc8d98973bdef2c56c74`

We plan to include scorer functionality together with scorer in main ParlAI repo in the near future.

To use `seq2seq_on_steroids` agent please do the following:
```bash
git clone git@github.com:facebookresearch/ParlAI.git
git clone git@github.com:uralik/beamybeam.git
cp -r beamybeam/parlai_external ParlAI/
cd ParlAI; python setup.py develop
``` 
After that you should be able to import SteroidSeq2seqAgent using this command:
```python
from parlai_external.agents.seq2seq_on_steroids.seq2seq_on_steroids import SteroidSeq2seqAgent
from parlai_external.agents.seq2seq_on_steroids.modules import SteroidSeq2seq
```

## Trained models
This is the model used for all experiments in the paper. Corresponding `.opt` file provides
all hyperparameters which were used during the training.

[seq2seq model](https://cims.nyu.edu/~kulikov/model_checkpoint.tar.gz)

Archive contains typical set of files needed in ParlAI to do any kinds of further tasks. Please see
ParlAI docs for further details.

To make a quick check to verify your model is running you can use eval script after paths
adjustments.

### Citation
Please use the following bib if you wish to cite our work:

```
@misc{kulikov2018importance,
    title={Importance of Search and Evaluation Strategies in Neural Dialogue Modeling},
    author={Ilia Kulikov and Alexander H. Miller and Kyunghyun Cho and Jason Weston},
    year={2018},
    eprint={1811.00907},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

