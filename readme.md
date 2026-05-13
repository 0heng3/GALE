# GALE

Clean Stage-II code for CIFAR-100-LT using the three GALE interfaces:

- `tailspec`: rare-class source signal generation.
- `ges`: gradient energy scaling.
- `comparb`: competence-based inference arbitration.

The default command reproduces the seed-42 setting:

```bash
python main_stage2.py 
```

Core code:

- `main_stage2.py`
- `losses/tailspec.py`
- `losses/bcl.py`
- `MultiBackward/GradFun/GES.py`
- `MultiBackward/ACCFun/CompArb.py`
- `train/validate.py`
- `models/resnet.py`
- `datasets/Cifar100LT.py`

