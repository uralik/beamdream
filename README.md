# Importance of a Search Strategy in Neural Dialogue Modelling

## Code

All the training and human evaluation was done using ParlAI framework. ParlAI is being actively
developed and we do not confirm our code to be working with master branch code. I

t should be working using ParlAI commit `5899c07934836d8757ebfc8d98973bdef2c56c74`

To use `seq2seq_on_steroids` agent please do the following:
```bash
git clone git@github.com:facebookresearch/ParlAI.git
git clone git@github.com:uralik/beamybeam.git ParlAI/parlai_external
cd ParlAI; python setup.py develop
``` 
After that you should be able to import SteroidSeq2seqAgent using this command:
```python
from parlai_external.agents.seq2seq_on_steroids.seq2seq_on_steroids import SteroidSeq2seqAgent
from parlai_external.agents.seq2seq_on_steroids.modules import SteroidSeq2seq
```



### Citation
Please use the following bib if you wish to cite our work:


