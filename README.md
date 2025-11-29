# Tiny Deep Feature Detection

Tiny feature detector trained using knowledge distillation using Superpoint model

<img width="901" height="128" alt="image" src="https://github.com/user-attachments/assets/dcc17ec2-2bca-4ec5-af40-49738c53bf87" />

- Dataset: [MS COCO](https://cocodataset.org/#home)
- Teacher Model: [Superpoint](https://huggingface.co/docs/transformers/main/model_doc/superpoint)
- Loss: Focal loss + Distillation loss (MSE(soft logits))
  
## Results

<img width="792" height="262" alt="image" src="https://github.com/user-attachments/assets/28ffedc1-ad0d-4548-a7c4-60d274022d68" />

HPatches benchmark [1]: Homography dataset
116 images, 5 homographies each

### Metric

<img width="694" height="237" alt="image" src="https://github.com/user-attachments/assets/71a74eb3-08b5-4092-a209-ad5288a77f2c" />

### 16.8x compression (77K params)

- Tested training thresholds (0.01-0.5): minimal effect
- Tested loss parameters (alpha, gamma, temp): found optimal config
- Result: Max 46 keypoints, 0.42 repeatability (65% of SuperPoint)

### 4.2x compression (306K params)

- Same hyperparameter sweep with 4× more capacity
- Result: Still capped at 46 keypoints, 0.44 repeatability
- Conclusion: Output resolution (60×80), not model capacity, is bottleneck

## Conclusion: We hit a spatial limit. Posssible fixes

- Reduce downsampling (3× or 2×) → 80×107 or 120×160 output
- Distillation loss between intermediate layers
- Dustbin channel (internal outlier rejection)

# References

[1] Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. (2017). HPatches: A benchmark and evaluation of handcrafted and learned local descriptors. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5173-5182).
