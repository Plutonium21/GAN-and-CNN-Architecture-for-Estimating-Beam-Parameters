
# BPM GAN Project

### Folder Structure

```
bpm_gan/
├── utils.py
├── create_bpm_dataset.py
├── gan_training.py
├── regressor.py
├── evaluate_gan.py
├── bpm_gan_dataset/
│   ├── images/
│   └── labels.csv
└── generated_samples/
```

### Run Order

1. Create dataset:
```bash
python create_bpm_dataset.py
```

2. Train GAN:
```bash
python gan_training.py
```

3. Train CNN Regressor:
```bash
python regressor.py
```

4. Evaluate GAN using regressor:
```bash
python evaluate_gan.py
```

---
Each image has only **one** type of noise (salt-and-pepper *or* Gaussian).
