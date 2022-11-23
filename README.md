# ASR project Kokorina Yulia AMI192

### WandB logger
https://wandb.ai/jakokorina/asr_project?workspace=user-jakokorina

### Wandb report
https://wandb.ai/jakokorina/asr_project/reports/ASR-Homework--VmlldzoyODAxODk5

## Installation guide

```
git clone https://github.com/jakokorina/asr_project_template.git -b branch2
pip install -r asr_project_template/requirements.txt
```

### Download checkpoint

```
pip install -U --no-cache-dir gdown --pre
gdown --id 1jWTaIXmVI1l6QZjBx-gzpJ2GNJnlLBrH
mv model_best.pth.gz asr_project_template/hw_asr/testing_data/
gzip -d asr_project_template/hw_asr/testing_data/model_best.pth.gz
```
### Testing on custom data
```
python3 asr_project_template/test.py \
   -c asr_project_template/hw_asr/testing_data/test_config.json \
   -r asr_project_template/hw_asr/testing_data/model_best.pth \
   -t asr_project_template/test_data \
   -o test_result.json \
   -b 5
```
### Testing on librispeech test data
```
python3 asr_project_template/test.py \
   -c asr_project_template/hw_asr/testing_data/config.json \
   -r asr_project_template/hw_asr/testing_data/model_best.pth \
   -o asr_project_template/test_result.json
```


### Testing with your file

Do not forget to use beam search! To do so:
1. Create variable text encoder:
`text_encoder = config.get_text_encoder()` or 

```
from hw_asr.text_encoder import CTCCharTextEncoder

text_encoder = CTCCharTextEncoder()
```

2. Then after applying model to batch do:

```
for i in range(len(batch["text"])):
   batch_beam_search_predicted = text_encoder.beam_search(
                            batch["log_probs"][i][:batch["log_probs_length"][i]].detach().cpu().numpy(),
                            beam_size=100
                        )
```

3. Do anything with predicted text.

## Model and checkpoints

All checkpoints are located in DataSphere. Since it's not an obvious thing how to download
many files from there, I'm ready to give any files if really need them.


