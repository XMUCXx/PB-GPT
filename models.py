import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.minGPT import GPT
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.init as init

class DecoderTransformer(nn.Module):
 
    def __init__(self, maxResidueLen = 512, d_model=768, nhead=12, dim_feedforward=3072, nlayers=12, dropout=0.1, n_dim=9, patch_size = 16, device=0):
        super(DecoderTransformer, self).__init__()

        self.device=device
        self.embedding=nn.Linear(patch_size * 9,d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (maxResidueLen // patch_size) + 1, d_model))
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout,
                                                 batch_first=True,
                                                 activation="gelu",
                                                 device=device
                                                 )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear=nn.Linear(d_model,n_dim * patch_size)
        
    def _generate_square_subsequent_mask(self, src, lengths):
        #[batch,seq_len]
        mask = torch.ones(src.size(0), src.size(1)) == 1
        for i in range(len(lengths)):
            lenth = lengths[i]
            for j in range(lenth):
                mask[i][j] = False
 
        return mask
 
    def forward(self, src, lengths):
        src_mask = self._generate_square_subsequent_mask(src,lengths).to(self.device)
        src = self.embedding(src)+self.pos_embedding[:, :lengths[0], :]
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.linear(output)
        #[b,l,n_dim * patch_size]
        return output
    
class Codebook(nn.Module):
    
    def __init__(self,patch_size = 16, num_codebook_vectors=50257):
        super(Codebook, self).__init__()
        self.n_dim = 9 * patch_size
        self.num_codebook_vectors= num_codebook_vectors
        
        self.embedding = nn.Embedding(num_codebook_vectors, self.n_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codebook_vectors, 1.0 / num_codebook_vectors)

    def forward(self, z, pkeep = 0):
        #z:[b,seq_len,n_dim * patch_szie] z_f:[b*seq_len,n_dim * patch_size]
        z_flattened = z.view(-1, self.n_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q - z)**2)

        if (pkeep > 0):
            mask = torch.bernoulli(pkeep * torch.ones(min_encoding_indices.shape, device=min_encoding_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            random_indices = torch.randint_like(min_encoding_indices, self.num_codebook_vectors)
            min_encoding_indices = mask * min_encoding_indices + (1 - mask) * random_indices
            z_q = self.embedding(min_encoding_indices).view(z.shape)

        return z_q, min_encoding_indices, loss
    
class RecModel(nn.Module):
    def __init__(self,    
                 maxResidueLen = 512,
                 d_model=768,      
                 nhead=12,
                 dim_feedforward=3072,
                 nlayers=12,
                 dropout=0., 
                 n_dim=9,
                 patch_size = 1,
                 num_codebook_vectors=50257,
                 device=0):
        super(RecModel, self).__init__()
        self.num_codebook_vectors=50257
        self.decoder = DecoderTransformer(maxResidueLen= maxResidueLen,d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,
                                   nlayers=nlayers,dropout=dropout,n_dim=9,patch_size = patch_size, device=device).to(device=device)
        self.codebook = Codebook(patch_size = patch_size, num_codebook_vectors=num_codebook_vectors).to(device=device)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x,lengths):
        codebook_mapping, codebook_indices, q_loss = self.codebook(x,0.0) # 0.0 mask
        decoded_x = self.decoder(codebook_mapping,lengths)

        return decoded_x, codebook_indices, q_loss

    def encode(self, x):
        codebook_mapping, codebook_indices, q_loss = self.codebook(x)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z,lengths):
        decoded_x = self.decoder(z,lengths)
        return decoded_x

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
    
class PBGPT(nn.Module):
    def __init__(self,      
                 maxResidueLen = 512,
                 RM_d_model=768, 
                 RM_nhead=12,
                 RM_dim_feedforward=3072,
                 RM_nlayers=12,
                 dropout=0., 
                 n_dim=9,
                 patch_size=1,
                 num_codebook_vectors=50257,
                 device=0,
                 checkpoint_path="RecModel_final.pt",
                 pkeep=1.0,
                 GPT_block_size=1024,
                 GPT_n_layers=24,
                 GPT_n_heads=16,
                 GPT_n_embd=1024,
                 ):
        super(PBGPT, self).__init__()
        self.pkeep=pkeep
        self.num_codebook_vectors=num_codebook_vectors

        model = RecModel(
                maxResidueLen,RM_d_model,RM_nhead,RM_dim_feedforward,RM_nlayers,dropout,
                n_dim,patch_size,num_codebook_vectors,device)
        model.load_checkpoint(checkpoint_path)
        model.eval()
        self.dmodel=model
        
        transformer_config = {
            "vocab_size": num_codebook_vectors+(maxResidueLen // patch_size) + 1,
            "block_size": GPT_block_size,
            "n_layer": GPT_n_layers,
            "n_head": GPT_n_heads,
            "n_embd": GPT_n_embd,
        }
        self.transformer = GPT(**transformer_config)

    @torch.no_grad()
    def encode_to_z(self, x):
        z, indices, _ = self.dmodel.encode(x)
        indices = indices.view(z.shape[0], -1)
        return z, indices

    @torch.no_grad()
    def z_to_backbone(self, indices,lengths):
        ix_to_vectors = self.dmodel.codebook.embedding(indices)
        backbone = self.dmodel.decode(ix_to_vectors,lengths)
        return backbone

    def forward(self, x, lengths):
        _, indices = self.encode_to_z(x)

        # for i,length in enumerate(lengths):
            # indices[i,length:]=self.num_codebook_vectors+1  #end prompt

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        sos_tokens = torch.zeros((x.shape[0], 2)) + self.num_codebook_vectors #start
        sos_tokens[:, 1] = torch.tensor(lengths) + self.num_codebook_vectors #length prompt
        sos_tokens = sos_tokens.long().to("cuda")

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = torch.cat((sos_tokens[:,1:], indices), dim=1)
        #target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature
            logits[:, self.num_codebook_vectors:] = -float("inf")

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_backbone(self, x,lengths):
        log = dict()

        _, indices = self.encode_to_z(x)

        sos_tokens = torch.zeros((x.shape[0], 2)) + self.num_codebook_vectors #start
        sos_tokens[:, 1] = torch.tensor(lengths) + self.num_codebook_vectors #length prompt
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_backbone(sample_indices,lengths)

        x_rec = self.z_to_backbone(indices,lengths)

        log["input"] = x
        log["rec"] = x_rec
        log["full_sample"] = full_sample

        return log