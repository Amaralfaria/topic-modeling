
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from utils.file import get_data_from_column

# ==========================================
# 1. Arquitetura Fiel da Rede Neural (ProdLDA)
# ==========================================
class OriginalProdLDANetwork(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden_size=100, dropout=0.6, alpha=1.0):
        super(OriginalProdLDANetwork, self).__init__()
        self.num_topics = num_topics
        
        # 1. Encoder Base (Input -> FC -> Softplus -> FC -> Softplus)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        # 2. Ramificação da Média (Figura 2: Mean -> mu)
        self.mu_layer = nn.Linear(hidden_size, num_topics)
        
        # 3. Ramificação da Variância (Figura 2: BN Layer -> Sigma)
        # CORREÇÃO: Adição explícita do BatchNorm1d exigido pelo artigo
        self.logvar_layer = nn.Sequential(
            nn.Linear(hidden_size, num_topics),
            nn.BatchNorm1d(num_topics, affine=False)
        )
        
        # 4. Decoder (Product of Experts)
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))
        self.beta_batchnorm = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop_theta = nn.Dropout(dropout)

        # 5. Aproximação de Laplace do Prior Dirichlet (Equação 6)
        prior_mean = math.log(alpha) - (1.0 / num_topics) * (num_topics * math.log(alpha))
        prior_var = (1.0 / alpha) * (1.0 - (2.0 / num_topics)) + (1.0 / (num_topics ** 2)) * (num_topics / alpha)
        
        self.register_buffer('prior_mean', torch.tensor([prior_mean] * num_topics, dtype=torch.float32))
        self.register_buffer('prior_var', torch.tensor([prior_var] * num_topics, dtype=torch.float32))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h) # Agora passa pelo Batch Norm
        
        z = self.reparameterize(mu, logvar)
        
        # Dropout na proporção de tópicos, conforme Seção 3.4
        theta = F.softmax(z, dim=-1)
        theta_dropped = self.drop_theta(theta)
        
        # Decoder Product of Experts (Equação da Seção 4.1)
        word_dist = F.softmax(self.beta_batchnorm(torch.matmul(theta_dropped, self.beta)), dim=-1)
        
        return word_dist, mu, logvar, theta
# ==========================================
# 2. Classe Wrapper Padrão
# ==========================================
class ProdLDAArtigo:
    def __init__(self, caminho_entrada, coluna_texto, coluna_tokens):
        self.caminho_entrada = caminho_entrada
        self.coluna_texto = coluna_texto
        self.coluna_tokens = coluna_tokens
        
        self.model = None
        self.vectorizer = None
        self.id2token = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data_tensor = None

    def fit(self, num_topicos, num_epochs=75, batch_size=64, learning_rate=2e-3, alpha=0.02):
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x)
        train_bow = self.vectorizer.fit_transform(self._get_tokens()).toarray()
        
        vocab = self.vectorizer.get_feature_names_out()
        self.id2token = dict(enumerate(vocab))
        vocab_size = len(vocab)
        
        self.train_data_tensor = torch.tensor(train_bow, dtype=torch.float32)
        dataset = TensorDataset(self.train_data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = OriginalProdLDANetwork(vocab_size=vocab_size, num_topics=num_topicos, alpha=alpha).to(self.device)
        
        # O artigo destaca a importância de um otimizador com alto momentum (beta1 > 0.8) 
        # para evitar que os componentes colapsem no início do treino.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.85, 0.999))
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                word_dist, mu, logvar, _ = self.model(x)
                
                # --- CÁLCULO DA LOSS ORIGINAL (Equação 7) ---
                
                # 1. Divergência KL entre N(mu, var) e o Prior Aproximado de Dirichlet N(prior_mean, prior_var)
                var = logvar.exp()
                term1 = var / self.model.prior_var
                term2 = (self.model.prior_mean - mu).pow(2) / self.model.prior_var
                term3 = self.model.prior_var.log() - logvar
                kl_loss = 0.5 * torch.sum(term1 + term2 - 1 + term3, dim=-1)
                
                # 2. Erro de Reconstrução (Log Likelihood da Multinomial)
                recon_loss = -(x * torch.log(word_dist + 1e-10)).sum(dim=-1)
                
                # Loss Total (Média do Batch)
                loss = (recon_loss + kl_loss).mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss / len(dataloader):.4f}")
                
        return self.model

    def get_document_topics(self):
        if self.model is None or self.train_data_tensor is None:
            raise ValueError("Execute .fit() primeiro.")
            
        self.model.eval()
        with torch.no_grad():
            x = self.train_data_tensor.to(self.device)
            _, _, _, theta = self.model(x)
            topic_predictions = theta.argmax(dim=-1).cpu().numpy()
            
        return topic_predictions.tolist()

    def get_topics(self, num_words=10):
        if self.model is None:
            raise ValueError("Execute .fit() primeiro.")
            
        beta_weights = self.model.beta.detach().cpu().numpy()
        
        topic_lists = []
        for topic_idx, topic_weights in enumerate(beta_weights):
            top_word_indices = topic_weights.argsort()[::-1][:num_words]
            top_words = [self.id2token[idx] for idx in top_word_indices]
            topic_lists.append((topic_idx, top_words))
            
        return topic_lists

    def _get_tokens(self):
        return list(get_data_from_column(self.caminho_entrada, self.coluna_tokens))

    def _get_text(self):
        return list(get_data_from_column(self.caminho_entrada, self.coluna_texto))