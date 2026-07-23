import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np
from utils.file import get_data_from_column

# ==========================================
# 1. Arquitetura Exata do Código Original (TensorFlow -> PyTorch)
# ==========================================
class OriginalTFProdLDANetwork(nn.Module):
    # Ajuste: dropout de 0.6 (equivale ao keep_prob=0.4 do TF) e alpha=1.0 (do código deles)
    def __init__(self, vocab_size, num_topics, hidden_size=100, dropout=0.6, alpha=1.0):
        super(OriginalTFProdLDANetwork, self).__init__()
        self.num_topics = num_topics
        
        # Encoder (Sem dropout interno, apenas no final)
        self.encoder_h1 = nn.Linear(vocab_size, hidden_size)
        self.encoder_h2 = nn.Linear(hidden_size, hidden_size)
        self.drop_encoder = nn.Dropout(dropout)
        
        # Ramificação da Média (Com Batch Norm explícito)
        self.mu_linear = nn.Linear(hidden_size, num_topics)
        self.mu_bn = nn.BatchNorm1d(num_topics)
        
        # Ramificação da Variância (Com Batch Norm explícito)
        self.logvar_linear = nn.Linear(hidden_size, num_topics)
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        
        # Decoder (Product of Experts)
        self.beta = nn.Parameter(torch.empty(num_topics, vocab_size))
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop_theta = nn.Dropout(dropout)

        # Aproximação do Prior baseada no TF original (a = 1.0)
        prior_mean = 0.0 # log(1) é 0
        prior_var = (1.0 / alpha) * (1.0 - (2.0 / num_topics)) + (1.0 / (num_topics ** 2)) * (num_topics / alpha)
        
        self.register_buffer('prior_mean', torch.tensor([prior_mean] * num_topics, dtype=torch.float32))
        self.register_buffer('prior_var', torch.tensor([prior_var] * num_topics, dtype=torch.float32))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Fluxo idêntico ao _recognition_network do TF
        l1 = F.softplus(self.encoder_h1(x))
        l2 = F.softplus(self.encoder_h2(l1))
        ldo = self.drop_encoder(l2)
        
        # Batch Norm aplicado na saída Linear
        mu = self.mu_bn(self.mu_linear(ldo))
        logvar = self.logvar_bn(self.logvar_linear(ldo))
        
        z = self.reparameterize(mu, logvar)
        
        # Fluxo idêntico ao _generator_network do TF
        theta = F.softmax(z, dim=-1)
        theta_dropped = self.drop_theta(theta)
        
        word_dist = F.softmax(self.beta_batchnorm(torch.matmul(theta_dropped, self.beta)), dim=-1)
        
        return word_dist, mu, logvar, theta
    
# ==========================================
# 2. Wrapper do Modelo
# ==========================================
class ProdLDA:
    def __init__(self, caminho_entrada, coluna_texto, coluna_tokens):
        self.caminho_entrada = caminho_entrada
        self.coluna_texto = coluna_texto
        self.coluna_tokens = coluna_tokens
        
        self.model = None
        self.vectorizer = None
        self.id2token = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data_tensor = None

    def fit(self, num_topicos, num_epochs=100, batch_size=200, learning_rate=0.002, beta1=0.99, beta2=0.999, alfa=1.0, dropout=0.6):
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x)
        train_bow = self.vectorizer.fit_transform(self._get_tokens()).toarray()
        
        vocab = self.vectorizer.get_feature_names_out()
        self.id2token = dict(enumerate(vocab))
        vocab_size = len(vocab)
        
        self.train_data_tensor = torch.tensor(train_bow, dtype=torch.float32)
        dataset = TensorDataset(self.train_data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        self.model = OriginalTFProdLDANetwork(vocab_size=vocab_size, num_topics=num_topicos, dropout=dropout, alpha=alfa).to(self.device)
        
        # Ajuste: beta1 fixado em 0.99 conforme o código original em TF
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                word_dist, mu, logvar, _ = self.model(x)
                
                # Divergência KL
                var = logvar.exp()
                term1 = var / self.model.prior_var
                term2 = (self.model.prior_mean - mu).pow(2) / self.model.prior_var
                term3 = self.model.prior_var.log() - logvar
                kl_loss = 0.5 * torch.sum(term1 + term2 - 1 + term3, dim=-1)
                
                # Erro de Reconstrução
                recon_loss = -(x * torch.log(word_dist + 1e-10)).sum(dim=-1)
                
                # Loss Total
                loss = (recon_loss + kl_loss).mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch) % 5 == 0:
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