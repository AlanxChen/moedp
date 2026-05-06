import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(
            self,
            num_experts,
            input_dim,
            output_dim,
            gate_dim,
            hidden_dim,
            top_k=2,
            dropout=0.1,
            boost=False,
            use_aux_loss=False,
            lambda_balance=1.0,
            lambda_entropy=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_aux_loss = use_aux_loss
        self.lambda_balance = lambda_balance
        self.lambda_entropy = lambda_entropy

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim))
            for _ in range(num_experts)])
        self.gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(gate_dim, num_experts)
            )
        ])

        # Initialize weights
        self._init_weights()

        if boost:
            self.boost = True
            self.top_k = top_k // 2
            self.working_experts = num_experts // 2
        else:
            self.boost = False
            self.top_k = top_k
            self.working_experts = num_experts

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def update_expert_num(self):
        if self.working_experts == self.num_experts:
            return
        self.working_experts += 4
        self.top_k += 1

    def forward(self, x, metrics=None, layer_num=0):
        batch_size = x.size(0)
        # Gate scores
        gate_scores_logits_ = self.gate[0](x)  # [batch_size, num_experts]
        if self.boost:
            gate_scores_logits = gate_scores_logits_[:, :self.working_experts]
        else:
            gate_scores_logits = gate_scores_logits_
        gate_scores = F.softmax(gate_scores_logits, dim=1)  # Softmax over experts for numerical stability

        # Top-k gate scores and indices
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=1)  # [batch_size, top_k]

        if metrics is not None:
            metrics[f'layer{layer_num}_top_{self.top_k}_scores'] = top_k_scores.detach()
            metrics[f'layer{layer_num}_top_{self.top_k}_indices'] = top_k_indices.detach()

        # Pass inputs through all experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts])  # [num_experts, batch_size, output_dim]
        expert_outputs = expert_outputs.permute(1, 0, 2)  # [batch_size, num_experts, output_dim]

        # Advanced indexing for selecting top-k expert outputs
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k).reshape(-1).to(x.device)
        selected_expert_outputs = expert_outputs[batch_indices, top_k_indices.reshape(-1)]  # [batch_size * top_k, output_dim]
        selected_expert_outputs = selected_expert_outputs.reshape(batch_size, self.top_k, self.output_dim)  # [batch_size, top_k, output_dim]

        # Scale the selected expert outputs by the corresponding gate scores
        scaled_expert_outputs = selected_expert_outputs * top_k_scores.unsqueeze(2)
        scaled_expert_outputs /= (top_k_scores.sum(dim=1, keepdim=True).unsqueeze(2) + 1e-9)  # Avoid division by zero

        # Sum the scaled expert outputs for the final output
        combined_output = scaled_expert_outputs.sum(dim=1)  # [batch_size, output_dim]
        if self.use_aux_loss:
            aux_loss, aux_loss_dict = moe_auxiliary_loss(
                gate_scores,
                top_k_indices,
                lambda_balance=self.lambda_balance,
                lambda_entropy=self.lambda_entropy
            )
            return combined_output, aux_loss, aux_loss_dict

        return combined_output, None, None


def moe_auxiliary_loss(gate_scores, top_k_indices, lambda_balance=1.0, lambda_entropy=0.1):
    # gate_scores: [B, E]
    # top_k_indices: [B, top_k]

    B, E = gate_scores.shape

    # Load Balancing Loss
    acc_probs = gate_scores.sum(dim=0)
    acc_freq = F.one_hot(top_k_indices, num_classes=E).float().sum(dim=[0, 1])

    switch_loss = E * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(acc_freq, p=1, dim=0)).sum()

    # Entropy Loss
    entropy = -(gate_scores * torch.log(gate_scores + 1e-9)).sum(dim=1).mean()

    # The policy multiplies this combined auxiliary loss by aux_loss_weight.
    auxiliary_loss = lambda_balance * switch_loss + lambda_entropy * entropy
    aux_loss_dict = {
        'switch_loss': switch_loss,
        "lambda_balance": lambda_balance,
        'entropy_loss': entropy,
        "lambda_entropy": lambda_entropy,
    }

    return auxiliary_loss, aux_loss_dict
