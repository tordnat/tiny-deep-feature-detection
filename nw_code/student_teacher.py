"""
Student-teacher training module used to train student on larger teacher model
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from tiny_keypoint_nn import TinyKeypointDetector
from torchvision.ops import sigmoid_focal_loss

class StudentTeacher(pl.LightningModule):
  def __init__(self, teacher_model, train_dataset, lr=1e-4, temperature=4.0, alpha=0.3, focal_alpha=0.75, focal_gamma=2.0, threshold=0.3):
    super().__init__()
    self.save_hyperparameters(ignore=['teacher_model', 'train_dataset'])

    # Freeze teacher weights
    self.teacher = teacher_model
    self.teacher.eval()
    for param in self.teacher.parameters():
      param.requires_grad = False
    
    # Student we want to train
    self.student = TinyKeypointDetector()
    self.init_student_weights()

    # Bias init
    prior_prob = 0.01  # 1% initial probability
    bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
    
    # Apply to final conv layer that outputs keypoint logits
    nn.init.constant_(self.student.head_out.bias, bias_value.item())

    self.lr = lr
    self.temperature = temperature 
    self.alpha = alpha
    self.focal_alpha = focal_alpha
    self.focal_gamma = focal_gamma
    self.threshold = threshold

    self.hparams.update({
        'lr': lr,
        'temperature': temperature,
        'alpha': alpha,
        'focal_alpha': focal_alpha,
        'focal_gamma': focal_gamma,
        'threshold': threshold
    })

  def init_student_weights(self):
      """Initialize weights following best practices for detection"""

      prior_prob = 0.01
      bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
      nn.init.constant_(self.student.head_out.bias, bias_value.item())
      
  def _generate_heatmap(self, pixel_values):
      """Generate sparse heatmap from SuperPoint keypoints"""
      outputs = self.teacher(pixel_values)
      
      batch_size = pixel_values.shape[0]
      h, w = 60, 80
      heatmaps = torch.zeros(batch_size, 1, h, w, device=pixel_values.device)
      
      keypoints = outputs.keypoints  # (B, N, 2) normalized [0,1]
      scores = outputs.scores        # (B, N)
      mask = outputs.mask.bool()     # (B, N) validity mask
      
      for b in range(batch_size):
          valid = mask[b]
          if not valid.any():
              continue
          
          kpts = keypoints[b][valid]
          scrs = scores[b][valid]
          
          # Convert to pixel coordinates
          kpts_pixel = (kpts * torch.tensor([w-1, h-1], device=kpts.device)).long()
          x_idx = kpts_pixel[:, 0].clamp(0, w-1)
          y_idx = kpts_pixel[:, 1].clamp(0, h-1)
          
          # Place scores at keypoint locations
          heatmaps[b, 0, y_idx, x_idx] = scrs
    
      return heatmaps

  def _get_teacher_soft_targets(self, pixel_values):

    with torch.no_grad():
      outputs = self.teacher(pixel_values)
      batch_size = pixel_values.shape[0]
      h, w = 60, 80

      heatmaps = torch.zeros(batch_size, 1, h, w, device=pixel_values.device)

      keypoints = outputs.keypoints
      scores = outputs.scores
      mask = outputs.mask.bool()

      for b in range(batch_size):
          valid = mask[b]
          if not valid.any():
            continue

          kpts = keypoints[b][valid]
          scrs = scores[b][valid]

          kpts_pixel = (kpts * torch.tensor([w-1, h-1], device=kpts.device)).long()
          x_idx = kpts_pixel[:, 0].clamp(0, w-1)
          y_idx = kpts_pixel[:, 1].clamp(0, h-1)
          
          # Use actual scores (soft targets)
          heatmaps[b, 0, y_idx, x_idx] = scrs

    return heatmaps 
  

  def training_step(self, batch, batch_idx):
    batch_size = batch.shape[0]
    # pixel_value, _ = batch

    teacher_soft = self._get_teacher_soft_targets(batch)
    student_logits = self.student(batch)

    # Distillation loss (soft targets with temperature)
    student_soft = torch.sigmoid(student_logits / self.temperature)
    distill_loss = F.mse_loss(student_soft, teacher_soft)
    
    # Hard loss with FOCAL LOSS for class imbalance
    hard_loss = sigmoid_focal_loss(
        student_logits,
        (teacher_soft > self.threshold).float(),
        alpha=self.focal_alpha,  # Balance pos/neg (0.25 emphasizes positives)
        gamma=self.focal_gamma,   # Focus on hard examples
        reduction='mean'
    )
    
    # Sparsity loss
    # sparsity_loss = torch.mean(student_soft[teacher_soft == 0])

    # Combined loss
    loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss # + 0.1 * sparsity_loss
    
    self.log('student_mean', student_logits.mean(), on_epoch=True)
    self.log('student_max', student_logits.max(), on_epoch=True)
    self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
    self.log('distill_loss', distill_loss, on_step=False, on_epoch=True)
    self.log('hard_loss', hard_loss, on_step=False, on_epoch=True)
    self.log('student_positive_ratio', (student_soft > 0.5).float().mean(), on_step=False, on_epoch=True)

    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch.shape[0]
    teacher_soft = self._get_teacher_soft_targets(batch)
    student_logits = self.student(batch)
    

    # Distillation loss (soft targets with temperature)
    student_soft = torch.sigmoid(student_logits / self.temperature)
    distill_loss = F.mse_loss(student_soft, teacher_soft)
    
    # Hard loss with FOCAL LOSS for class imbalance
    hard_loss = sigmoid_focal_loss(
        student_logits,
        (teacher_soft > 0.3).float(),
        alpha=0.25,  # Balance pos/neg (0.25 emphasizes positives)
        gamma=2.0,   # Focus on hard examples
        reduction='mean'
    )
    
    # Combined loss
    loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

    self.log('val_loss', loss, prog_bar=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    config = {'optimizer': optimizer, 'lr_scheduler': {'scheduler':scheduler, 'monitor':'train_loss'}}
    return config