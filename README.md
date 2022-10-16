# ASR project Kokorina Yulia AMI192

## Installation guide

```shell
git clone https://github.com/jakokorina/asr_project_template.git -b branch2
cd asr_project_template
pip install -r ./requirements.txt
```

```download checkpoint
pip install -U --no-cache-dir gdown --pre
gdown --id 1jWTaIXmVI1l6QZjBx-gzpJ2GNJnlLBrH
mv model_best.pth.gz asr_project_template/hw_asr/testing_data/
gzip -d asr_project_template/hw_asr/testing_data/model_best.pth.gz
```

```testing on custom data
python test.py \
   -c hw_asr/testing_data/test_config.json \
   -r hw_asr/testing_data/model_best.pth \
   -t test_data \
   -o test_result.json \
```

```testing on librespeech test data
python test.py \
   -c hw_asr/testing_data/test_config.json \
   -r hw_asr/testing_data/model_best.pth \
   -o test_result.json \
```

## Model and checkpoints

All checkpoints are located in DataSphere. Since it's not an obvious thing how to download
a lot of files from there, I'm ready to give any files if you tell me how.


