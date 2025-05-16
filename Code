import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import csv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import kenlm

# ----------- Preprocessing -----------

def preprocess_audio(file_path, sr=22050, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    y = librosa.effects.preemphasis(y)
    y = librosa.util.normalize(y)
    y = vad(y)
    y = noise_reduction(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def vad(y, top_db=20):
    intervals = librosa.effects.split(y, top_db=top_db)
    return np.concatenate([y[start:end] for start, end in intervals])

def noise_reduction(y):
    noise_sample = y[:10000]
    return y - np.mean(noise_sample)

# ----------- Metadata Transcript Extraction -----------

def extract_transcripts_from_metadata(metadata_path, wavs_dir):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            file_id, transcript = row[0], row[1]
            with open(os.path.join(wavs_dir, f'{file_id}.txt'), 'w') as txt_file:
                txt_file.write(transcript.strip())

# ----------- Dataset Loader -----------

def load_dataset(path):
    X, y = [], []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            mfcc = preprocess_audio(os.path.join(path, file))
            label_path = os.path.join(path, file.replace(".wav", ".txt"))
            if not os.path.exists(label_path):
                continue
            with open(label_path, 'r') as f:
                label = f.read().strip()
            X.append(mfcc)
            y.append(label)
    return X, y

# ----------- Dataset Class -----------

class SpeechDataset(Dataset):
    def __init__(self, X, y, max_len=100):
        self.X = X
        self.y = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        label = self.y[idx][:self.max_len].ljust(self.max_len)
        y_tensor = torch.tensor([ord(c) for c in label], dtype=torch.long)
        return x_tensor, y_tensor

# ----------- Acoustic Model -----------

class AcousticRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output)

# ----------- Seq2Seq Decoder -----------

class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_len=100):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.max_len = max_len

    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), 1, hidden.size(2)).to(x.device)
        outputs = []
        for _ in range(self.max_len):
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            attn_weights = torch.bmm(encoder_outputs, output.transpose(1, 2)).squeeze(-1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            concat = torch.cat((context, output), dim=2)
            attn_applied = self.attn(concat)
            out = self.fc(attn_applied.squeeze(1))
            outputs.append(out)
            decoder_input = output
        return torch.stack(outputs, dim=1)

# ----------- Beam Search with KenLM -----------

def beam_search_decoder(logits, beam_width=5, lm_path=None):
    if lm_path is None:
        # fallback to greedy
        return torch.argmax(logits, dim=1).tolist()
    model = kenlm.Model(lm_path)
    sequences = [([], 0.0)]
    for t in logits:
        candidates = []
        for seq, score in sequences:
            for i in range(len(t)):
                candidates.append((seq + [i], score - float(t[i])))
        sequences = sorted(candidates, key=lambda x: x[1])[:beam_width]
    best = max(sequences, key=lambda x: model.score(" ".join(map(str, x[0]))))
    return best[0]

# ----------- Evaluation -----------

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)

# ----------- Training -----------

def collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys_tensor = torch.stack(ys)
    return xs_padded, ys_tensor

def train_models(X, y, input_dim, hidden_dim, output_dim, batch_size=4, epochs=10):
    dataset = SpeechDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = AcousticRNN(input_dim, hidden_dim, output_dim).cuda()
    decoder = Seq2SeqDecoder(output_dim, hidden_dim, output_dim).cuda()
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        decoder.train()
        for x_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            logits = model(x_batch)
            preds = decoder(logits)
            loss = criterion(preds.view(-1, output_dim), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    with open("acoustic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("seq2seq_decoder.pkl", "wb") as f:
        pickle.dump(decoder, f)
    return model, decoder

# ----------- Inference -----------

def inference(audio_path, use_kenlm=False, lm_path=None):
    with open("acoustic_model.pkl", "rb") as f:
        acoustic_model = pickle.load(f)
    with open("seq2seq_decoder.pkl", "rb") as f:
        decoder = pickle.load(f)

    acoustic_model.cuda().eval()
    decoder.cuda().eval()

    features = preprocess_audio(audio_path)
    x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = acoustic_model(x_tensor)
        decoded_logits = decoder(logits)

    decoded_logits_np = decoded_logits.squeeze(0).cpu().numpy()

    if use_kenlm and lm_path:
        predicted_ids = beam_search_decoder(decoded_logits_np, lm_path=lm_path)
    else:
        predicted_ids = np.argmax(decoded_logits_np, axis=1).tolist()

    hypothesis = "".join([chr(c) for c in predicted_ids if 32 <= c < 128])
    return hypothesis

# ----------- Main -----------

if __name__ == "__main__":
    dataset_path = "/content/dataset/LJSpeech-1.1/wavs"
    metadata_path = "/content/dataset/LJSpeech-1.1/metadata.csv"

    print("Extracting transcripts...")
    extract_transcripts_from_metadata(metadata_path, dataset_path)

    print("Loading dataset...")
    X, y = load_dataset(dataset_path)

    max_char = max([max(ord(c) for c in label) for label in y])
    output_dim = max_char + 1
    print(f"Max char ordinal: {max_char}, output_dim set to: {output_dim}")

    model_rnn, decoder = train_models(X, y, input_dim=13, hidden_dim=128, output_dim=output_dim)

    # Example inference (change path as needed)
    test_file = "/content/dataset/LJSpeech-1.1/wavs/LJ001-0001.wav"
    print("Performing inference...")
    hypothesis = inference(test_file, use_kenlm=False)
    print(f"Predicted text: {hypothesis}")
