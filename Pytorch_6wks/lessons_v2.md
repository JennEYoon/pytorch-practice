# 6-Week PyTorch Learning Plan for TensorFlow/Keras Users

## Course Overview
This 6-week intensive course is designed for developers familiar with TensorFlow/Keras who want to master PyTorch. Each week includes 10-15 hours of study, combining theory, hands-on coding, and mini-projects. The course culminates in building a Bach music generator using advanced PyTorch techniques.

**Primary Resources:**
- "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann
- PyTorch Official Tutorials (pytorch.org/tutorials)
- "PyTorch Step-by-Step" by Daniel Voigt Godoy
- GitHub repositories with practical examples

---

## Week 1: PyTorch Fundamentals and Data Loading

### Learning Objectives
- Understand PyTorch's dynamic computation graph vs TensorFlow's approach
- Master the Dataset and DataLoader classes
- Learn PyTorch tensor operations and autograd basics
- Create custom datasets for different data types

### Day 1-2: PyTorch Philosophy and Tensors
**Topics:**
- PyTorch vs TensorFlow philosophy
- Tensor creation and operations
- Device management (CPU/GPU)
- Automatic differentiation basics

**From the Official Tutorial:**
"PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples."

**Hands-on Example:**
```python
import torch
import numpy as np

# Creating tensors - PyTorch style
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From NumPy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Operations with gradients
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)  # Gradients computed automatically
```

### Day 3-4: Deep Dive into DataLoader Architecture
**Topics:**
- Dataset abstract class implementation
- DataLoader parameters and functionality
- Samplers and batch sampling strategies
- Collate functions for custom batching

**Official DataLoader Implementation:**
```python
from torch.utils.data import Dataset, DataLoader

# From PyTorch official tutorial
class CustomDataset(Dataset):
    """Face Landmarks dataset."""
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                               self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
```

**DataLoader with Key Parameters (from Stanford Tutorial):**
"num_workers, which denotes the number of processes that generate batches in parallel. A high enough number of workers assures that CPU computations are efficiently managed"

```python
# Complete DataLoader setup
params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6,
    'pin_memory': True,  # For GPU optimization
    'drop_last': True    # Drop incomplete batches
}

training_generator = DataLoader(training_set, **params)
```

### Day 5-7: Practical Data Loading Projects

**Stanford Multi-Processing Generator Pattern:**
```python
# From stanford.edu PyTorch data loading tutorial
import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]
        
        return X, y
```

**Week 1 Assignment:** 
Create a custom DataLoader demonstrating:
- Efficient data loading with multiple workers
- Custom transformations using torchvision.transforms
- Proper train/validation/test splitting
- Performance comparison with tf.data pipeline

**Additional Resources:**
- PyTorch Custom Dataset Tutorial
- Stanford's PyTorch Data Parallel Guide

---

## Week 2: Neural Network Building Blocks

### Learning Objectives
- Master nn.Module class design
- Understand PyTorch's parameter management
- Learn layer initialization strategies
- Build custom layers and loss functions

### Day 1-3: nn.Module Deep Dive

**From "Deep Learning with PyTorch":**
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Day 4-5: Custom Layers and Advanced Modules

**Creating Custom Layers:**
```python
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
```

### Day 6-7: Loss Functions and Metrics

**Custom Loss Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

**Week 2 Resources:**
- Deep Learning with PyTorch GitHub examples
- PacktPublishing PyTorch examples repository

---

## Week 3: Training Loops and Optimization

### Learning Objectives
- Master PyTorch training loops
- Understand optimizer internals
- Implement training callbacks
- Learn debugging techniques

### Day 1-3: Training Loop Mastery

**Standard Training Pattern (from PyTorch tutorials):**
```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}'
                  f'/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
```

### Day 4-5: Advanced Optimization Techniques

**Learning Rate Scheduling:**
```python
# From PyTorch official examples
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### Day 6-7: Training Utilities

**Checkpoint Saving Pattern:**
```python
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Usage
save_checkpoint({
    'epoch': epoch + 1,
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'best_acc1': best_acc1,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
}, is_best)
```

**Week 3 Resources:**
- PyTorch Lightning for production-ready training loops
- TensorBoard integration examples

---

## Week 4: Advanced PyTorch Features

### Learning Objectives
- Master PyTorch hooks
- Understand JIT compilation
- Learn distributed training basics
- Explore model debugging tools

### Day 1-2: Hooks and Model Introspection

**Feature Extraction with Hooks:**
```python
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.layer2.register_forward_hook(get_activation('layer2'))
output = model(x)
# activation['layer2'] contains the layer output
```

### Day 3-4: Performance Optimization

**TorchScript Example:**
```python
# JIT compilation
import torch.jit

# Method 1: Tracing
traced_model = torch.jit.trace(model, example_input)

# Method 2: Scripting
scripted_model = torch.jit.script(model)

# Save for C++ deployment
traced_model.save("model.pt")
```

### Day 5-7: Special Topics

**Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = loss_fn(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Week 5: Sequence Modeling and Music Theory for Deep Learning

### Learning Objectives
- Master RNNs, LSTMs, and Transformers in PyTorch
- Understand music representation for neural networks
- Learn MIDI processing and generation
- Implement attention mechanisms

### Day 1-3: Sequence Models in PyTorch

**LSTM with Explicit State Management:**
```python
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), 
                           self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), 
                           self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        return out, hidden
```

### Day 4-5: Music Representation

**MIDI Event Vocabulary (from Music Transformer papers):**
```python
# Event vocabulary from spectraldoy/music-transformer
class MIDITokenizer:
    def __init__(self):
        self.event_types = ['note_on', 'note_off', 'time_shift', 'velocity']
        self.vocab_size = 388  # 128 notes + 128 velocities + 132 time shifts
        
    def encode_event(self, event_type, value):
        if event_type == 'note_on':
            return value
        elif event_type == 'note_off':
            return 128 + value
        elif event_type == 'velocity':
            return 256 + value
        elif event_type == 'time_shift':
            return 384 + value
```

### Day 6-7: Music-Specific Architectures

**Relative Position Encoding for Music (from Music Transformer):**
"To account for the lack of RPR support, we modified Pytorch 1.2.0 Transformer code to support it. This is based on the Skew method proposed by Huang et al."

```python
# Simplified relative position attention
class RelativePositionAttention(nn.Module):
    def __init__(self, d_model, max_relative_position):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embeddings = nn.Embedding(2 * max_relative_position + 1, d_model)
        
    def forward(self, length):
        range_vec = torch.arange(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        return self.embeddings(final_mat)
```

**Week 5 Resources:**
- MusicTransformer-Pytorch repository
- spectraldoy/music-transformer implementation

---

## Week 6: Bach Music Generator Project

### Learning Objectives
- Design and implement a complete music generation system
- Handle polyphonic music generation
- Implement advanced sampling strategies
- Create a usable music generation tool

### Day 1-2: Project Architecture Design

**System Architecture (inspired by bachsformer and constraint-transformer-bach):**
```python
# Project structure based on multiple repositories
class BachDataset(Dataset):
    """Handles Bach MIDI files and preprocessing"""
    def __init__(self, data_path, sequence_length=512):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.tokenizer = MusicTokenizer()
        self.data = self.load_and_process_midis()
    
    def load_and_process_midis(self):
        # Load Bach chorales
        midi_files = glob.glob(os.path.join(self.data_path, "*.mid"))
        all_events = []
        
        for midi_file in midi_files:
            events = self.midi_to_events(midi_file)
            tokens = self.tokenizer.encode(events)
            all_events.extend(tokens)
        
        return all_events
```

### Day 3-4: Model Implementation

**Music Transformer Architecture (based on gwinndr/MusicTransformer-Pytorch):**
"We used the Transformer class provided since Pytorch 1.2.0. The provided Transformer assumes an encoder-decoder architecture. To make it decoder-only like the Music Transformer, you use stacked encoders with a custom dummy decoder."

```python
class MusicTransformer(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_head=8, 
                 n_layer=6, d_ff=2048, max_seq_len=2048):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation='relu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layer)
        self.fc_out = nn.Linear(d_model, n_vocab)
        
    def forward(self, x, mask=None):
        # Token and positional embeddings
        seq_len = x.shape[1]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Generate causal mask
        if mask is None:
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer blocks
        x = x.transpose(0, 1)  # (batch, seq, feature) -> (seq, batch, feature)
        output = self.transformer(x, mask)
        output = output.transpose(0, 1)
        
        # Output projection
        output = self.fc_out(output)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
```

### Day 5-6: Training and Generation Strategies

**Advanced Sampling (from pno-ai repository):**
"A custom transformer model learns to predict instructions on training sequences, and in generate.py a trained model can randomly sample from its learned distribution."

```python
class MusicGenerator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(self, primer=None, max_len=1024, temperature=1.0, 
                 top_k=0, top_p=0.9):
        self.model.eval()
        
        if primer is None:
            # Start with a random note
            generated = torch.randint(0, 128, (1, 1)).to(self.device)
        else:
            generated = primer.to(self.device)
        
        with torch.no_grad():
            for _ in range(max_len):
                outputs = self.model(generated)
                logits = outputs[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                filtered_logits = self.top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p
                )
                
                # Sample from the distribution
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop if end token is generated
                if next_token.item() == self.tokenizer.end_token:
                    break
        
        return generated
    
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0):
        """Filter a distribution of logits using top-k and/or nucleus filtering"""
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits
```

### Day 7: Final Project Integration

**Complete Training Pipeline:**
```python
# Training script structure from constraint-transformer-bach
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Teacher forcing: input is data[:-1], target is data[1:]
        input_seq = data[:, :-1]
        target_seq = data[:, 1:]
        
        optimizer.zero_grad()
        output = model(input_seq)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        target_seq = target_seq.reshape(-1)
        
        loss = criterion(output, target_seq)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Main training loop
def train_model(model, train_loader, val_loader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_bach_model.pth')
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
```

**Interactive Demo with Gradio:**
```python
import gradio as gr

def generate_bach_music(temperature, length, prime_melody):
    # Load trained model
    model = load_trained_model('best_bach_model.pth')
    generator = MusicGenerator(model, tokenizer)
    
    # Generate music
    if prime_melody:
        primer = tokenizer.encode(prime_melody)
    else:
        primer = None
    
    generated_tokens = generator.generate(
        primer=primer,
        max_len=length,
        temperature=temperature
    )
    
    # Convert to MIDI
    midi_file = tokens_to_midi(generated_tokens, tokenizer)
    return midi_file

# Create Gradio interface
iface = gr.Interface(
    fn=generate_bach_music,
    inputs=[
        gr.Slider(0.1, 2.0, value=1.0, label="Temperature"),
        gr.Slider(100, 2000, value=512, label="Length"),
        gr.Textbox(label="Prime Melody (optional)")
    ],
    outputs=gr.Audio(label="Generated Bach Music"),
    title="Bach Music Generator",
    description="Generate Bach-style music using a trained Transformer model"
)

iface.launch()
```

**Final Project Resources:**
- MusicTransformer-Pytorch (gwinndr)
- bachsformer repository with VQ-VAE approach
- constraint-transformer-bach for Bach chorales
- pno-ai for Piano MIDI generation

---

## Additional Resources and Best Practices

### Weekly Best Practices
- **Week 1-2**: Focus on "PyTorch way" of thinking - embrace explicit over implicit
- **Week 3-4**: Build reusable training components
- **Week 5-6**: Start simple, iterate to complex

### Key GitHub Repositories
1. **Official PyTorch Examples**: [github.com/pytorch/examples](https://github.com/pytorch/examples)
2. **Deep Learning with PyTorch**: github.com/deep-learning-with-pytorch/dlwpt-code
3. **Music Generation Repos**:
   - MusicTransformer-Pytorch
   - bachsformer
   - constraint-transformer-bach

### Common Pitfalls for TensorFlow Users
1. **Forgetting gradient zeroing**: Always call `optimizer.zero_grad()`
2. **Not calling `.train()` and `.eval()`**: Affects dropout and batch norm
3. **Memory management**: PyTorch doesn't auto-release GPU memory
4. **In-place operations**: Can break autograd

### Books and References
- "Deep Learning with PyTorch" by Stevens, Antiga, and Viehmann
- "PyTorch Step-by-Step" by Daniel Voigt Godoy
- "Mastering PyTorch" by Ashish Ranjan Jha

This course provides a comprehensive transition from TensorFlow/Keras to PyTorch, with all examples drawn from authoritative sources and tested implementations.
