import torch
import torch.nn as nn
import numpy as np

# ===============================
# 1. Load Dataset
# ===============================

with open(r"C:\Users\edlap\Documents\FInal year Project\Raja\Codsoft\handwritten.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Vocabulary Size:", vocab_size)

# Character mappings
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode text
encoded = np.array([char_to_idx[c] for c in text])

# ===============================
# 2. Create Training Data
# ===============================

seq_length = 50
inputs = []
targets = []

for i in range(len(encoded) - seq_length):
    inputs.append(encoded[i:i+seq_length])
    targets.append(encoded[i+1:i+seq_length+1])

inputs = torch.tensor(np.array(inputs))
targets = torch.tensor(np.array(targets))

# ===============================
# 3. RNN Model
# ===============================

class CharRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super(CharRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):

        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# ===============================
# 4. Initialize Model
# ===============================

hidden_size = 128
model = CharRNN(vocab_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# ===============================
# 5. Train Model
# ===============================

epochs = 20
batch_size = 64

for epoch in range(epochs):

    hidden = model.init_hidden(batch_size)

    for i in range(0, len(inputs) - batch_size, batch_size):

        x = inputs[i:i+batch_size]
        y = targets[i:i+batch_size]

        optimizer.zero_grad()

        output, hidden = model(x, hidden.detach())

        loss = criterion(
            output.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ===============================
# 6. Text Generation
# ===============================

def generate_text(model, start_text="once ", length=200):

    model.eval()

    chars = list(start_text.lower())
    input_seq = torch.tensor([[char_to_idx.get(c,0) for c in chars]])

    hidden = model.init_hidden(1)

    generated = start_text

    for _ in range(length):

        output, hidden = model(input_seq, hidden)

        probs = torch.softmax(output[0, -1], dim=0).detach().numpy()

        char_idx = np.random.choice(len(probs), p=probs)

        next_char = idx_to_char[char_idx]

        generated += next_char

        input_seq = torch.tensor([[char_idx]])

    return generated


# ===============================
# 7. Generate Text
# ===============================

print("\nGenerated Text:\n")

print(generate_text(model, "once ", 300))