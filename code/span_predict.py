import torch
import torch.nn as nn


class SpanPredict(nn.Module):
    def __init__(self, num_spans: int, input_dims: int):
        super(SpanPredict, self).__init__()
        self.num_spans = num_spans
        self.input_dims = input_dims
        self.pq_softmax = nn.Softmax(1)
        self.layer_p = nn.Linear(self.input_dims, self.num_spans)
        self.layer_q = nn.Linear(self.input_dims, self.num_spans)

    def forward(self, inputs, token_mask):
        p_tilde = self.pq_softmax(self.layer_p(inputs))
        q_tilde = self.pq_softmax(self.layer_q(inputs))

        p = torch.cumsum(p_tilde, 1)

        q = torch.flip(q_tilde, [1])
        q = torch.cumsum(q, 1)
        q = torch.flip(q, [1])

        r_parts = []
        for ind in range(self.num_spans):
            r_part = p[:, :, ind] * q[:, :, ind]  # B, S
            r_part = r_part / (torch.sum(r_part, 1).unsqueeze(1) + 1e-8)
            r_part = r_part * token_mask  # B, S
            r_parts.append(r_part.unsqueeze(2))  # B,S,1
        r = torch.concat(r_parts, 2)  # B, S, J
        return r
