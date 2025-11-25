import torch
import matplotlib.pyplot as plt

def visualize_results(model, image, processor):
    """Visualize teacher vs student heatmaps side-by-side"""
    model.eval()
    
    # Prepare image
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs['pixel_values']
    
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
        model = model.cuda()
    
    with torch.no_grad():
        teacher_heatmap = model._generate_heatmap(pixel_values)[0, 0].cpu().numpy()
        
        # Student (apply sigmoid for visualization)
        student_logits = model.student(pixel_values)
        student_heatmap = torch.sigmoid(student_logits)[0, 0].cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(teacher_heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Teacher (SuperPoint)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(student_heatmap, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Student (TinyKeypointDetector)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    return fig
