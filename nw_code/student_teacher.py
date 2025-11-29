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
  def __init__(self, teacher_model, train_dataset, lr=1e-4, temperature=4.0, alpha=0.3, 
               focal_alpha=0.75, focal_gamma=2.0, threshold=0.3):
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
    nn.init.constant_(self.student.head_out.bias, bias_value.item())

    self.lr = lr
    self.temperature = temperature 
    self.alpha = alpha
    self.focal_alpha = focal_alpha
    self.focal_gamma = focal_gamma
    self.threshold = threshold


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
          
          heatmaps[b, 0, y_idx, x_idx] = scrs

    return heatmaps 
  
  def _compute_loss(self, batch):
    """Shared loss computation for train/val/test"""
    teacher_soft = self._get_teacher_soft_targets(batch)
    student_logits = self.student(batch)

    # Distillation loss
    student_soft = torch.sigmoid(student_logits / self.temperature)
    distill_loss = F.mse_loss(student_soft, teacher_soft) * (self.temperature ** 2)
    
    # Hard loss with focal loss
    hard_loss = sigmoid_focal_loss(
        student_logits,
        (teacher_soft > self.threshold).float(),
        alpha=self.focal_alpha,
        gamma=self.focal_gamma,
        reduction='mean'
    )
    
    # Combined loss
    loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
    
    return loss, distill_loss, hard_loss, student_logits

  def training_step(self, batch, batch_idx):
    loss, distill_loss, hard_loss, student_logits = self._compute_loss(batch)
    
    # Logging
    self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
    self.log('distill_loss', distill_loss, on_step=False, on_epoch=True)
    self.log('hard_loss', hard_loss, on_step=False, on_epoch=True)
    self.log('student_mean', student_logits.mean(), on_epoch=True)
    self.log('student_max', student_logits.max(), on_epoch=True)
    
    return loss

  def validation_step(self, batch, batch_idx):
    loss, distill_loss, hard_loss, _ = self._compute_loss(batch)
    
    self.log('val_loss', loss, prog_bar=True)
    self.log('val_distill_loss', distill_loss)
    self.log('val_hard_loss', hard_loss)
    
    return loss
  
  def test_step(self, batch, batch_idx):
    """Evaluate on test set using same loss"""
    loss, distill_loss, hard_loss, _ = self._compute_loss(batch)
    
    self.log('test_loss', loss, prog_bar=True)
    self.log('test_distill_loss', distill_loss)
    self.log('test_hard_loss', hard_loss)
    
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    return {
        'optimizer': optimizer, 
        'lr_scheduler': {
            'scheduler': scheduler, 
            'monitor': 'val_loss'
        }
    }