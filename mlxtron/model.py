import mlx.nn as nn

class Llama(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # sanity check 
        assert config.hidden_size % config.num_attention_heads==0
        assert config.num_attention_heads % config.num_key_value_heads==0 

        # params
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads 
        self.head_dim = self.hidden_size//self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config

        # modules
        self.embedding = Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config,layer_idx = i) for i in range(self.num_layers)])
        self.final_proj = FinalProjection(self.hidden_size, self.vocab_size, bias=False)
        RMSNorm = LlamaRMSNorm if os.getenv('FLASH_ATTEN', '1') != '1' else TritonRMSNorm
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.reset_parameters()

