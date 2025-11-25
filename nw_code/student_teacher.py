"""
Student-teacher training module used to train student on larger teacher model
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from tiny_keypoint_nn import TinyKeypointDetector

class StudentTeacher(pl.LightningModule):
  def __init__(self, teacher_model, train_dataset, lr=1e-4):
    super().__init__()
    self.save_hyperparameters(ignore=['teacher_model', 'train_dataset'])

    # Freeze teacher weights
    self.teacher = teacher_model
    self.teacher.eval()
    for param in self.teacher.parameters():
      param.requires_grad = False
    
    # Student we want to train
    self.student = TinyKeypointDetector()
    self.lr = lr


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

  def training_step(self, batch, batch_idx):
    batch_size = batch.shape[0]

    # Use cached teacher heatmap (no teacher inference!)
    teacher_heatmap = self._generate_heatmap(batch)

    student_logits = self.student(batch)
    loss = F.binary_cross_entropy_with_logits(student_logits, teacher_heatmap)
    
    self.log('train_loss', loss, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch.shape[0]
    teacher_heatmap = self._generate_heatmap(batch)
    student_logits = self.student(batch)
    
    loss = F.binary_cross_entropy_with_logits(student_logits, teacher_heatmap)
    
    self.log('val_loss', loss, prog_bar=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    config = {'optimizer': optimizer, 'lr_scheduler': {'scheduler':scheduler, 'monitor':'train_loss'}}
    return config