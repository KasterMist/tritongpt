import torch
import torch.nn as nn

# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (wte): Embedding(50257, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     (h): ModuleList(
#       (0-11): 12 x GPT2Block(
#         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (attn): GPT2Attention(
#           (c_attn): Conv1D(nf=2304, nx=768)
#           (c_proj): Conv1D(nf=768, nx=768)
#           (attn_dropout): Dropout(p=0.1, inplace=False)
#           (resid_dropout): Dropout(p=0.1, inplace=False)
#         )
#         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (mlp): GPT2MLP(
#           (c_fc): Conv1D(nf=3072, nx=768)
#           (c_proj): Conv1D(nf=768, nx=3072)
#           (act): NewGELUActivation()
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=50257, bias=False)
# )

class GPT2Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token Embedding
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)  # Positional Embedding
        self.dropout = nn.Dropout(config.embd_pdrop)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        embeddings = token_embeds + position_embeds
        return self.dropout(embeddings)

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // config.n_head
        self.scale = self.head_dim ** -0.5

        self.c_attn = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer("bias", torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(hidden_size, dim=-1)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Apply causal mask
        attn_weights = attn_weights.masked_fill(self.bias[:, :, :seq_length, :seq_length] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        # Final projection
        return self.resid_dropout(self.c_proj(attn_output))
    
class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.intermediate_size = config.n_embd * 4
        self.c_fc = nn.Linear(self.hidden_size, self.intermediate_size)
        self.c_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.summary_first_dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.hidden_size)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.embeddings = GPT2Embeddings(config)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_head)])
        self.ln_f = nn.LayerNorm(self.hidden_size)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：lm_head.weight 与 wte.weight 共享
        self.lm_head.weight = self.transformer.embeddings.wte.weight

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits


from transformers import AutoTokenizer
from transformers import AutoConfig
from safetensors.torch import load_file
config = AutoConfig.from_pretrained("openai-community/gpt2")
# 实例化自定义模型
model = GPT2LMHeadModel(config)

# model.load_state_dict(torch.load("/home/letian/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors", weights_only=True))
print(model)
# 加载预训练权重（自动匹配命名）
# weight_file = "/home/letian/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
# weights = load_file(weight_file)
# print(len(weights))
# model.load_state_dict(weights)
# print("load finish")
# 设置为评估模式
model.eval()