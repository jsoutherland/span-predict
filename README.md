# SpanPredict
Pytorch implementation of the paper:  [SpanPredict: Extraction of Predictive Document Spans with Neural Attention](https://aclanthology.org/2021.naacl-main.413/)

## Project Purpose
I'm implementing the architecture described in the paper as an educational exercise, to make sure I understand it.
I'm making the code available as I could only find two implementations that were publicly available so far. 
One was in Tensorflow and the other was in Pytorch.

There is a lot of data wrangling and pretrained model code in the repo so far.  I think if someone was interested in re-using this code 
they would just be interested in:
* `code/span_predict.py` - For the `SpanPredict` module
* `code/jsd.py` - For the JSD loss
* `code/imdb_model.py` - To see how to use the `SpanPredict` module

Parts of `code/main.py` might be useful to reference, to see how to handle training.


## Project Status
I've reproduced the CNN baseline model and SpanPredict models and trained them both against the IMDB dataset.
I haven't gotten very far into inspecting the resulting spans, but the performance numbers closely match those from
the paper. I'm updating this as I have time as a side project.


## Install
You'll need to:
* download the IMDB dataset 
* download the glove embeddings
* Install python requirements from `requirements.txt`


## Training
Steps:
* run the `code/prep_data.py` script to generate `imdb_train_test.csv` from the IMDB dataset.
* run `code/main.py` to train a model. 
  * The boolean `SPAN_MODEL` setting at the top of the script can be used to switch between the baseline CNN model (`False`) or SpanPredict model (`True`)
  * TODO: the new latter half of `code/main.py` may not work yet with the baseline CNN model


## Span Inspection
* TODO: I'm still working on this step
* The beginning of this is `code/inspect_model.py`


## Deviations from the paper
* I currently use the entire ~400k GloVe vocabulary (I'm not computing token frequency to allow for token prioritization yet.)
* The learning settings (JSD ramp-up and epoch length) may be currently adjusted for faster training in an IMDB dataset-specific way.
* I'm experimenting with which weights to freeze (embedding weights, conv layers, etc.) and when in the training process


## References
* [SpanPredict: Extraction of Predictive Document Spans with Neural Attention](https://aclanthology.org/2021.naacl-main.413/)
* SpanPredict Paper [supplementary code](https://aclanthology.org/attachments/2021.naacl-main.413.OptionalSupplementaryCode.zip)
* [PIP: Physical Interaction Prediction via Mental Simulation with Span Selection](https://dl.acm.org/doi/abs/10.1007/978-3-031-19833-5_24)
* PIP Paper [repo](https://github.com/SamsonYuBaiJian/pip)
