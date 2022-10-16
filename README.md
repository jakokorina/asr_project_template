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
```
### Testing on librispeech test data
```
python3 asr_project_template/test.py \
   -c asr_project_template/hw_asr/testing_data/config.json \
   -r asr_project_template/hw_asr/testing_data/model_best.pth \
   -o asr_project_template/test_result.json \
```

## Model and checkpoints

All checkpoints are located in DataSphere. Since it's not an obvious thing how to download
many files from there, I'm ready to give any files if you tell me how.


