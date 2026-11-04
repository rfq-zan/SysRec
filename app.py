import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import time
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="East Java Smart Tourism", layout="wide")
st.title("🗺️ East Java Smart Tourism Recommender")
st.markdown("**Context-Aware Adaptive Fusion System (BGE-M3 + E5)**")

class Gate(nn.Module):
    def __init__(self, d_in, h=32, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

class GateFeatures:
    def __init__(self, blocks):
        self.blocks = blocks
    def _num(self, Fd):
        parts = []
        if 'qpp' in self.blocks: parts.append(Fd['qpp'])
        if 'emb' in self.blocks: parts.append(self.pca.transform(Fd['qemb']))
        return np.hstack(parts) if parts else None
    def transform(self, Fd):
        parts = []
        num = self._num(Fd)
        if num is not None: parts.append(self.scaler.transform(num))
        if 'ctx' in self.blocks: parts.append(Fd['ctx'])
        return np.hstack(parts).astype(np.float32)

def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)

def topk_idx(S, k):
    return np.argsort(-S, axis=1, kind='stable')[:, :k]

def qpp_block(S):
    srt = -np.sort(-S, 1); z = -np.sort(-zscore(S), 1)
    p = np.exp(z[:, :10] - z[:, :1]); p /= p.sum(1, keepdims=True)
    return np.c_[srt[:, 0], srt[:, 0] - srt[:, 1], srt[:, :10].mean(1), srt[:, :10].std(1), S.std(1),
                 z[:, 0], z[:, 0] - z[:, 9], -(p * np.log(p + 1e-12)).sum(1)]

def agreement(SB, SE, k=10):
    a, b_ = topk_idx(SB, k), topk_idx(SE, k)
    jac = np.array([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b_)])
    ra = np.argsort(np.argsort(SB, 1), 1).astype(float); rb = np.argsort(np.argsort(SE, 1), 1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    rho = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.c_[jac, rho]

def build_feats(SB, SE, QB, QE, seasons, texts):
    CTX_IDX = {'hujan': 0, 'kemarau': 1, 'none': 2}
    ctx = np.zeros((len(SB), 3), np.float32)
    ctx[np.arange(len(SB)), [CTX_IDX[s] for s in seasons]] = 1
    qlen = np.log1p([len(t.split()) for t in texts])
    return dict(qpp=np.c_[qpp_block(SB), qpp_block(SE), agreement(SB, SE), qlen].astype(np.float32),
                qemb=np.hstack([QB, QE]).astype(np.float32), ctx=ctx)

def fuse(lam, ZB, ZE):
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float32).reshape(-1, 1), (ZB.shape[0], 1))
    return lam * ZB + (1 - lam) * ZE

@st.cache_resource
def load_system():
    df = pd.read_excel('koreksi_dataset.xlsx')
    df['wisata'] = df['wisata'].fillna('lainnya').astype(str).str.strip().str.lower()
    df['text'] = df['text_original'].fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[df['text'].str.split().str.len() >= 3]
    df = df.drop_duplicates(['destinasi', 'text'])
    
    dest = (df.groupby('destinasi')
              .agg(wisata=('wisata', lambda s: s.mode().iat[0]),
                   musim=('rekomendasi_cuaca', lambda s: s.mode().iat[0]),
                   n_ulasan=('text', 'size'),
                   rating_mean=('rating', 'mean'))
              .reset_index())
    
    dest['cocok_hujan'] = dest['musim'].str.contains('Penghujan')
    dest_id = {d: i for i, d in enumerate(dest['destinasi'])}
    
    df['did'] = df['destinasi'].map(dest_id)
    df = df.sort_values('did', kind='stable').reset_index(drop=True)
    offsets = np.r_[0, np.cumsum(dest['n_ulasan'].values)]
    
    device = 'cpu'
    # rb_t = torch.from_numpy(np.load('cache_embedding/bge_corpus_3583fe1c97.npy')).to(device, torch.float32)
    # re_t = torch.from_numpy(np.load('cache_embedding/e5_corpus_8ded02df08.npy')).to(device, torch.float32)
    
    b1 = np.load('cache_embedding/bge_corpus_3583fe1c97_pt1.npy')
    b2 = np.load('cache_embedding/bge_corpus_3583fe1c97_pt2.npy')
    b3 = np.load('cache_embedding/bge_corpus_3583fe1c97_pt3.npy')
    rb_t = torch.from_numpy(np.concatenate([b1, b2, b3])).to(device, torch.float32)

    e1 = np.load('cache_embedding/e5_corpus_8ded02df08_pt1.npy')
    e2 = np.load('cache_embedding/e5_corpus_8ded02df08_pt2.npy')
    e3 = np.load('cache_embedding/e5_corpus_8ded02df08_pt3.npy')
    re_t = torch.from_numpy(np.concatenate([e1, e2, e3])).to(device, torch.float32)

    bge = SentenceTransformer('BAAI/bge-m3', device=device)
    e5 = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    
    with open('output_fusions/gate_features.pkl', 'rb') as f:
        fb_final = pickle.load(f)
        
    gates_final = []
    for i in range(3):
        gate = Gate(d_in=38) 
        gate.load_state_dict(torch.load(f'output_fusions/gate_model_{i}.pt', map_location=device, weights_only=True))
        gate.eval()
        gates_final.append(gate)
        
    return df, dest, offsets, rb_t, re_t, bge, e5, fb_final, gates_final

with st.spinner("Initializing AI Models and Loading Dataset... (This takes a moment on startup)"):
    df, dest, offsets, rb_t, re_t, bge, e5, fb_final, gates_final = load_system()

def aggregate_scores(S, topm=5):
    out = torch.empty((S.shape[0], len(dest)), device=S.device, dtype=torch.float32)
    for d in range(len(dest)):
        seg = S[:, offsets[d]:offsets[d + 1]]
        v = seg.topk(min(topm, seg.shape[1]), dim=1).values
        out[:, d] = torch.nanmean(v, dim=1)
    return out.cpu().numpy()

st.sidebar.header("🔍 Search Criteria")
query = st.sidebar.text_area("Describe your ideal destination:", placeholder="Example: Pantai yang tenang untuk keluarga...")
musim = st.sidebar.selectbox("Season Context", ["none", "hujan", "kemarau"])
filter_musim = st.sidebar.checkbox("Apply Hard Season Filter?", help="Removes destinations entirely if they don't fit the season.")

if st.sidebar.button("Find Recommendations", type="primary"):
    if query:
        st.write("---")
        start_time = time.time()
        
        with st.spinner("Generating Adaptive Fusion Recommendations..."):
            qB = bge.encode([query], convert_to_numpy=True).astype(np.float32)
            qE = e5.encode(["query: " + query], convert_to_numpy=True).astype(np.float32)
            
            qB_t = torch.from_numpy(qB).to('cpu')
            qE_t = torch.from_numpy(qE).to('cpu')
            sB = aggregate_scores((qB_t @ rb_t.T).float())
            sE = aggregate_scores((qE_t @ re_t.T).float())
            
            feats = build_feats(sB, sE, qB, qE, [musim], [query])
            X = fb_final.transform(feats)
            lam = float(np.mean([gate(torch.tensor(X, dtype=torch.float32)).detach().numpy() for gate in gates_final], axis=0)[0])
            
            s = fuse(lam, zscore(sB), zscore(sE))[0]
            if filter_musim and musim == 'hujan':
                not_rain = ~dest['cocok_hujan'].values
                s = np.where(not_rain, -np.inf, s)
                
            top = np.argsort(-s, kind='stable')[:10]
            latency = time.time() - start_time
            
            st.subheader("🌟 Top 10 Recommendations")
            st.info(f"⏱️ Inference time: {latency*1000:.0f} ms | 🧠 **Dynamic AI Weights:** BGE-M3 ({lam:.3f}) / E5 ({1-lam:.3f})")
            
            simB = (rb_t @ qB_t.T).float().cpu().numpy().flatten()
            simE = (re_t @ qE_t.T).float().cpu().numpy().flatten()
            sim = lam * simB + (1 - lam) * simE
            
            for i, d_idx in enumerate(top):
                destination_name = dest.iloc[d_idx]['destinasi']
                rating = dest.iloc[d_idx]['rating_mean']
                
                with st.expander(f"#{i+1} - {destination_name} (⭐ {rating:.1f})"):
                    st.write(f"**Category:** {dest.iloc[d_idx]['wisata'].title()} | **Season Suitability:** {dest.iloc[d_idx]['musim']}")
                    
                    best_review_idx = offsets[d_idx] + int(np.argmax(sim[offsets[d_idx]:offsets[d_idx + 1]]))
                    st.markdown(f"> *\"{df.iloc[best_review_idx]['text_original']}\"*")
    else:
        st.warning("Please enter a destination description first.")