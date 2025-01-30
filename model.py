import inspect
import torch
from dataclasses import dataclass
from torch.nn import functional as F

@dataclass
class ModelParams:
    context_length: int = 512
    vocab_size: int = 50257
    num_blocks: int = 12
    num_heads: int = 12
    d_model: int = 768
    assert d_model % num_heads == 0, "Number of heads must divide model dimension"
    head_dim: int = d_model // num_heads
    dropout_rate: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    

class AttentionHead(torch.nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.key = torch.nn.Linear(params.d_model,params.head_dim) # Linear, some use conv1D
        self.query = torch.nn.Linear(params.d_model,params.head_dim) # Linear, some use conv1D
        self.value = torch.nn.Linear(params.d_model,params.head_dim) # Linear, some use conv1D
        self.dropout = torch.nn.Dropout(params.dropout_rate)
        self.device = params.device
        

    def forward(self,x):
        k = self.dropout(self.key(x))
        q = self.dropout(self.query(x))
        v = self.dropout(self.value(x))
        _,T,dk = k.shape
        
        # dot_product_attention = q @ k.transpose(2,1) # MatMul
        # scaled_dot_product_attention = dot_product_attention / torch.sqrt(torch.tensor(dk)) # Scale
        # masked_scaled_dot_product_attention = scaled_dot_product_attention.masked_fill((torch.tril(torch.ones(T,T)) == 0).to(self.device),float("-inf")) # Mask
        # soft_masked_scaled_dot_product_attention = self.dropout(torch.softmax(masked_scaled_dot_product_attention,dim=-1)) # Softmax
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return y
    
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self,params: ModelParams):
        super().__init__()
        self.heads = torch.nn.ModuleList([AttentionHead(params=params) for _ in range(params.num_heads)])
        self.proj = torch.nn.Linear(params.d_model,params.d_model)

    def forward(self,X):
        out = torch.cat([h(X) for h in self.heads],dim=-1) # Concat
        return self.proj(out) # Linear (proj)

class PositionWiseFeedforward(torch.nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.c_fc = torch.nn.Linear(in_features=params.d_model,out_features=params.d_model*4)
        self.gelu = torch.nn.GELU(approximate="tanh") # GELU activation function from https://arxiv.org/pdf/1606.08415.pdf
        self.c_proj = torch.nn.Linear(in_features=params.d_model*4,out_features=params.d_model)
        self.c_proj.DRAGON_SCALE = 1.0
        
    def forward(self,X):
        X = self.c_fc(X)
        X = self.gelu(X)
        out = self.c_proj(X)
        return out

class DecoderBlock(torch.nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.ln_1 = torch.nn.LayerNorm(params.d_model) # MMSA layer norm
        self.attn = MultiHeadSelfAttention(params) #Masked Multi-Head Self-Attention
        self.ln_2 = torch.nn.LayerNorm(params.d_model) # PWFFN layer norm
        self.mlp = PositionWiseFeedforward(params) # Positionwise Feedforward Network

    def forward(self,X):
        X = X + self.attn(self.ln_1(X))
        out = X + self.mlp(self.ln_2(X))
        return out
        
class BeeHummingBirdLM(torch.nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(params.vocab_size, params.d_model),
            wpe = torch.nn.Embedding(params.context_length, params.d_model),
            h = torch.nn.ModuleList([DecoderBlock(params) for _ in range(params.num_blocks)]),
            ln_f = torch.nn.LayerNorm(params.d_model)
        ))
        self.lm_head = torch.nn.Linear(params.d_model, params.vocab_size, bias=False)
        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight # weight tying
    
    # init parameters
    def init_weights(self):
        self.apply(self._init_weights)
        return self
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            std = 0.02
            if hasattr(module, "DRAGON_SCALE"):
                std *= (2 * self.params.num_blocks) ** -0.5
            torch.torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, X: torch.Tensor):
        _,T = X.shape
        assert T <= self.params.context_length, f"Input sequence length {T} is greater than maximum context length {self.params.context_length}"
        pos = torch.arange(0, T, dtype=torch.long, device=X.device).unsqueeze(0) # (1,T)
        pos_embed = self.transformer.wpe(pos) # (1,T,C)
        text_embed = self.transformer.wte(X) # (B,T,C)
        X = text_embed + pos_embed # (B,T,C)
        for block in self.transformer.h:
            X = block(X)
        X = self.transformer.ln_f(X) # (B,T,C)
        logits = self.lm_head(X) # (B,T,vocabulary_size)
        return logits
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
        
    def generate(self, current_context, max_new_tokens):
        assert len(current_context) < self.params.context_length, f"Input sequence length {len(current_context)} is greater than maximum context length {self.params.context_length}"
        for _ in range(max_new_tokens): # current_context is (B, T) array of indices in the current context
            current_context_cond = current_context[:, -self.params.context_length:] # crop current_context to the last context_size tokens
            logits = self(current_context_cond) # get the predictions
            logits = logits[:, -1, :] # focus only on the last time step : becomes (B, C)
            probs = torch.nn.functional.softmax(logits, dim=-1) # apply softmax to get probabilities : becomes (B, C)
            next_token = torch.multinomial(probs, num_samples=1) # sample from the distribution : becomes (B, 1)
            current_context = torch.cat((current_context, next_token), dim=1) # append sampled index to the running sequence : becomes (B, T+1)
        return current_context
    
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    