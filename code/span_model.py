import torch
import torch.nn as nn


class IMDBSpan(nn.Module):
    def __init__(self, embedding_matrix, num_spans=3):
        super(IMDBSpan, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=embedding_matrix.shape[0], embedding_dim=embedding_matrix.shape[1])
        # self.embedding.requires_grad_(False)

        num_embed_dims = embedding_matrix.shape[1]

        self.n_filters = 50
        self.num_spans = num_spans
        self.convs = nn.ParameterList([
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=2, padding='same', stride=1),
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=3, padding='same', stride=1),
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=5, padding='same', stride=1)
        ])

        self.K = len(self.convs) * self.n_filters
        self.pq_softmax = nn.Softmax(1)
        self.layer_p = nn.Linear(self.K, self.num_spans)
        self.layer_q = nn.Linear(self.K, self.num_spans)
        self.layer_z = nn.Linear(self.K, 1)
        self.relu = nn.ReLU()
        self.cnn_dropout = nn.Dropout(0.5)
        self.layer_out = nn.Linear(150, 1)
        self.tanh = nn.Tanh()

    def forward(self, token_ids, token_mask):
        x = self.embedding(token_ids)  # B, S, D
        x = x * token_mask.unsqueeze(2)
        # print('token_mask', token_mask.shape) # B, S
        x = x.permute(0, 2, 1)  # B,D,S
        conv_outs = [conv(x) for conv in self.convs]
        E = torch.concat(conv_outs, 1)  # B, K, S
        E = E.permute(0, 2, 1)  # B, S, K
        E = self.relu(E)
        E = self.cnn_dropout(E)
        E = E * token_mask.unsqueeze(2)  # B,S,K * B,S,1 = B,S,K

        p_tilde = self.pq_softmax(self.layer_p(E))
        q_tilde = self.pq_softmax(self.layer_q(E))

        p = torch.cumsum(p_tilde, 1)

        q = torch.flip(q_tilde, [1])
        q = torch.cumsum(q, 1)
        q = torch.flip(q, [1])

        r_parts = []
        for ind in range(self.num_spans):
            r_part = p[:, :, ind] * q[:, :, ind]  # B, S
            r_part = r_part / (torch.sum(r_part, 1).unsqueeze(1) + 1e-8)
            r_part = r_part * token_mask # B, S
            r_parts.append(r_part.unsqueeze(2))  # B,S,1
        r = torch.concat(r_parts, 2)  # B, S, J

        M_parts = []
        for ind in range(self.num_spans):
            m = E * r[:, :, ind].unsqueeze(2)  # B,S,K
            # m = torch.mean(m, 1).unsqueeze(2)  # B, K, 1  Mean pooling
            m = torch.max(m, 1).values.unsqueeze(2)  # Max pooling
            m = m.permute(0, 2, 1)  # B,1,K
            M_parts.append(m)
        M = torch.concat(M_parts, 1)  # B,J,K

        z = self.layer_z(M)  # B, J, 1
        z = self.tanh(z)

        y_pred = torch.sum(z, 1)  # B, 1
        return y_pred, r
