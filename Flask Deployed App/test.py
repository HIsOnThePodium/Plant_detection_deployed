import torch
import torch.quantization

# Load pre-trained model
model = torch.load('plant_disease_model_1_latest.pt')

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model, 'quantized_model.pt')
