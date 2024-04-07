# Stega4NeRF

This is the official code for "Stega4NeRF: Cover Selection Steganography for Neural Radiance Fields"

# Running the code

This code requires Python 3. 

You can find the pretrained models at `Stega4NeRF\modelD=1.pt`.

---


To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```
After training for 200k iterations (~8 hours on a single 2080 Ti), you can find the following video at `Stega4NeRF\logs\blender_paper_lego1\blender_paper_lego1_spiral_200000_rgb.mp4`.

---
To render images from different viewpoints
```
python render_new-viewpoints-images.py 
```

---


To train a message extractor (train one-to-one mapping of secret viewpoint image to secret Messages by overfitting):
Take D=1 as an example
```
python train_extractor.py 
```

After training for 2000 iterations (~27 s on a single 2080 Ti), you can find the following model at `Stega4NeRF\modelD=1.pt`.

---

To train a classification model (Implement a disguise for message extractor):
```
python train_cifar10.py 
```
After training for 500 iterations (~5 hours on a single 2080 Ti), you can find the following model at `Stega4NeRF\modelD=1.pt`.

---

To test new perspective synthesized images (Use the correct extractor key and trained modelD=1.pt)  :
```
python test_secret.py 
```
---

To test hybrid model performance (Use trained modelD=1.pt)  :
```
python test_cifar10.py 
```
---

The model file “model.pt” and the data in the “data” and “logs” folders can be downloaded from Baidu Netdisk.
```
The link is: https://pan.baidu.com/s/1s4HAhMQhgBiwjhhHiMMJig
The extraction code is: o98s
```
---
# Acknowledgements

[NeRF](https://paperswithcode.com/paper/nerf-representing-scenes-as-neural-radiance#code) models are used to implement Stega4NeRF. 
