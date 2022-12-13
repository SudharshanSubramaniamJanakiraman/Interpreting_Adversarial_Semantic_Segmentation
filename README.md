# Evaluating & Interpreting Adversarial Robustness of Semantic Segmentation for Autonomous Driving

Goal : Understand the effect of Adversarial attacks on State of the Art Semantic Segmentation Model
```
.
├── README.md
├── adv_dag.py
├── conf
│   ├── cam.yaml
│   └── dag.yaml
├── datamodule.py
├── dataset.py
├── lightning.py
├── loss.py
├── prediction_viz.py
├── pspnet.py
├── run.py
├── seg_grad_sam.py
└── utils.py
```

INSTALL THE REQUIREMENTS
```
pip install -r requirements.txt
```

TO TRAIN THE SEMANTIC SEGMENTATAION MODEL
```
python run.py
```
TO INTEPRET SEMNATIC SEGMENTATION OUTPUT
```
git clone https://github.com/jacobgil/pytorch-grad-cam.git 
cd pytorch-grad-cam
mv pytorch_grad_cam ~/Interpreting_Adversarial_Semantic_Segmentation/.
```
then
```
python seg_grad_sam.py
```
For Generating Adversarial Attacks
```
python adv_dag.py
```

