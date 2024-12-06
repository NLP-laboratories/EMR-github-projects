import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertModel
from torchcrf import CRF
from sklearn.metrics.pairwise import cosine_similarity
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # 使用 x.size(0) 确保编码的大小与输入兼容
        return x + self.encoding[:, :x.size(0), :].to(x.device)

class CGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CGCN, self).__init__()
        self.positional_encoding = PositionalEncoding(in_dim)
        self.gcn1 = GCNConv(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.gcn2 = GCNConv(out_dim, out_dim)

    def forward(self, x, edge_index):
        # 添加位置编码
        x = self.positional_encoding(x)  # x 形状为 (num_nodes, in_dim)
        x = self.gcn1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        return x

class Model(nn.Module):
    def __init__(self, ltp):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.bn_lstm = nn.BatchNorm1d(2 * HIDDEN_SIZE)
        
        self.cgcn_seq = CGCN(EMBEDDING_DIM, HIDDEN_SIZE)  # 字符级顺序图
        self.cgcn_dep = CGCN(EMBEDDING_DIM, HIDDEN_SIZE)  # 词级句法依存图
        
        self.seq_feats_linear = nn.Linear(HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        self.dep_feats_linear = nn.Linear(HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        
        self.fusion_linear = nn.Linear(3 * 2 * HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, word_embeddings, mask):
        lstm_out, _ = self.lstm(word_embeddings)
        batch_size, seq_len, hidden_size = lstm_out.size()
        lstm_out = lstm_out.contiguous().view(-1, hidden_size)
        lstm_out = self.bn_lstm(lstm_out)
        lstm_out = lstm_out.view(batch_size, seq_len, hidden_size)
        return lstm_out

    def _sample_subgraph(self, edge_index, embeddings=None, valid_mask=None, hop=1, sample_type='similarity', top_k=3):
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        if hop == 1:
            return edge_index

        elif hop == 2:
            if sample_type == 'similarity' and embeddings is not None:
                if valid_mask is not None:
                    embeddings = embeddings[valid_mask]
                num_nodes, _ = embeddings.shape

                similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
                similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float).to(embeddings.device)

                actual_top_k = min(top_k, num_nodes - 1)

                sampled_edges = []
                for node in range(num_nodes):
                    top_k_indices = torch.topk(similarity_matrix[node], k=actual_top_k + 1).indices[1:]

                    most_similar_idx = node
                    found_similar = False

                    for idx in top_k_indices:
                        if idx.item() != node:
                            most_similar_idx = idx.item()
                            found_similar = True
                            break
                
                    if not found_similar:
                        most_similar_idx = node
            
                    sampled_edges.append((node, most_similar_idx))

                sampled_edges = torch.tensor(sampled_edges, dtype=torch.long).to(embeddings.device)

                full_sampled_edges = []
                for i, (src, dst) in enumerate(edge_index):
                    if valid_mask[src] and valid_mask[dst]:
                        full_sampled_edges.append(sampled_edges[i].tolist())
                    else:
                        full_sampled_edges.append([-1, -1])

                full_sampled_edges_tensor = torch.tensor(full_sampled_edges, dtype=torch.long).contiguous().to(embeddings.device)

                return full_sampled_edges_tensor
            else:
                return edge_index

    def forward(self, input, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2,
                word_vectors, input_texts, dependency_masks=None, word_vectors_masks=None, return_logits=False):
        device = input.device

        # 获取BERT的词嵌入
        word_embeddings = self.bert(input, mask).last_hidden_state
        lstm_out = self._get_lstm_feature(word_embeddings, mask)

        word_vectors = [vec.to(device) for vec in word_vectors]
        dependency_masks = [dep_mask.to(device) for dep_mask in dependency_masks]
        word_vectors_masks = [mask.to(device) for mask in word_vectors_masks]

        # 对字符级顺序图进行特征提取
        seq_feats = [
            self.cgcn_seq(word_embeddings[idx], torch.tensor(seq_edge, dtype=torch.long).t().contiguous().to(device))
            for idx, seq_edge in enumerate(sequence_edges)
        ]

        # 对词级句法依存图进行特征提取
        dep_feats_hop1 = [
            self.cgcn_dep(word_vectors[idx].clone().detach(), dep_edge.t().to(device))
            for idx, dep_edge in enumerate(dependency_edges_hop1)
        ]
        dep_feats_hop2 = [
            self.cgcn_dep(word_vectors[idx].clone().detach(), dep_edge.t().contiguous().to(device))
            for idx, dep_edge in enumerate(dependency_edges_hop2)
        ]

        # 合并依存图的特征
        dep_feats = [
            torch.cat([hop1, hop2], dim=-1) * dep_mask.float().unsqueeze(-1)
            for hop1, hop2, dep_mask in zip(dep_feats_hop1, dep_feats_hop2, dependency_masks)
        ]
        
        seq_feats = [self.seq_feats_linear(feat) for feat in seq_feats]

        # 对单词向量应用掩码
        word_vectors = [
            vec * word_mask.unsqueeze(-1) for vec, word_mask in zip(word_vectors, word_vectors_masks)
        ]

        # 合并所有特征
        combined_feats = [
            self.fusion_linear(torch.cat([lstm_out[idx], seq_feats[idx], dep_feats[idx]], dim=-1))
            for idx in range(len(seq_feats))
        ]
        
        logits = self.linear(torch.stack(combined_feats))
        
        if return_logits:
            return logits
        decoded_tags = self.crf.decode(logits, mask)
        return decoded_tags

    def loss_fn(self, input, target, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2, word_vectors, input_texts, dependency_masks=None, word_vectors_masks=None):
        logits = self.forward(input, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2, word_vectors, input_texts, dependency_masks, word_vectors_masks, return_logits=True)
        loss = -self.crf(logits, target, mask=mask, reduction='mean')
        return loss
