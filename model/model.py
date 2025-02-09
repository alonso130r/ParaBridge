import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import sent_tokenize

# Ensure that the NLTK punkt tokenizer is available.
nltk.download('punkt', quiet=True)

###############################################
# Dynamic rotary aggregator that can combine an arbitrary number
# of sentence embeddings into a single paragraph embedding.
###############################################
class DynamicRotaryAggregator(nn.Module):
    """
    Aggregates an arbitrary number (N) of sentence embeddings (each of dimension D)
    into a single vector of dimension D. For each sentence embedding, a learnable
    rotation is applied to a designated "rotary" subset of features (rotary_dim). 
    Then a gating mechanism produces a weight per sentence, and a weighted average is computed.
    """
    def __init__(self, hidden_size, rotary_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        # If rotary_dim is not provided, use half of hidden_size (or all, per design)
        self.rotary_dim = rotary_dim if rotary_dim is not None else hidden_size
        # Learnable angle factor (a vector of length rotary_dim) used to compute rotation angles.
        self.angle_factor = nn.Parameter(torch.randn(self.rotary_dim))
        # A small gating layer that computes a scalar weight for each sentence embedding.
        self.gate = nn.Linear(hidden_size, 1)
    
    def forward(self, sentence_embeddings):
        """
        Args:
            sentence_embeddings: a tensor of shape [N, hidden_size], where N is the number of sentences.
        Returns:
            A tensor of shape [hidden_size] that aggregates all N sentence embeddings.
        """
        N = sentence_embeddings.size(0)
        device = sentence_embeddings.device
        
        # Compute a weight for each sentence.
        # weights: [N, 1]
        weights = torch.sigmoid(self.gate(sentence_embeddings))
        
        # Create a tensor of positions [0, 1, ..., N-1] (shape: [N, 1])
        positions = torch.arange(N, device=device).unsqueeze(1).float()
        
        # Compute rotation angles: shape [N, rotary_dim]
        angles = positions * self.angle_factor  # broadcasting multiplication
        
        # Compute cosine and sine for each angle
        cos_vals = torch.cos(angles)  # [N, rotary_dim]
        sin_vals = torch.sin(angles)  # [N, rotary_dim]
        
        # Split each sentence embedding into two parts:
        #   - the first rotary_dim dimensions (to be rotated)
        #   - the remaining dimensions (unchanged)
        x_rot = sentence_embeddings[:, :self.rotary_dim]  # [N, rotary_dim]
        x_rest = sentence_embeddings[:, self.rotary_dim:]  # [N, hidden_size - rotary_dim]
        
        # (A standard RoPE approach that pairs dimensions.)
        half_dim = self.rotary_dim // 2
        x_even = x_rot[:, 0::2]
        x_odd = x_rot[:, 1::2]
        cos_even = cos_vals[:, :half_dim]
        sin_even = sin_vals[:, :half_dim]
        rot_even = x_even * cos_even - x_odd * sin_even
        rot_odd = x_odd * cos_even + x_even * sin_even
        x_rotated = torch.stack([rot_even, rot_odd], dim=-1).reshape(N, self.rotary_dim)
        
        # Reassemble the full embedding.
        rotated_embeddings = torch.cat([x_rotated, x_rest], dim=1)  # [N, hidden_size]
        
        # Aggregate by computing the weighted sum over sentences.
        weighted_sum = (rotated_embeddings * weights).sum(dim=0)
        normalization = weights.sum() + 1e-9
        aggregated = weighted_sum / normalization  # [hidden_size]
        return aggregated

###############################################
# Standard aggregator (max-pool).
###############################################
class MaxPoolAggregator(nn.Module):
    """
    Aggregates a list of sentence embeddings using max pooling.
    """
    def __init__(self):
        super().__init__()

    def forward(self, sentence_embeddings_list, num_sentences, encoder_hidden_size):
        padded_embeddings = pad_sequence(sentence_embeddings_list, batch_first=True)
        batch_size, _, _ = padded_embeddings.shape
        aggregated = []
        for i in range(batch_size):
            valid_outputs = padded_embeddings[i, :num_sentences[i], :]
            if valid_outputs.size(0) > 0:
                pooled = valid_outputs.max(dim=0)[0]
            else:
                pooled = torch.zeros(encoder_hidden_size, device=padded_embeddings.device)
            aggregated.append(pooled)
        return torch.stack(aggregated, dim=0)

###############################################
# Alignment Modules
###############################################
class BaseAlignmentModule(nn.Module):
    """
    Base class for alignment modules that transform paragraph embeddings
    into a series of embeddings for the decoder (soft prompts or similar).
    """
    def __init__(self, input_dim, output_dim, prompt_length, use_eos=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prompt_length = prompt_length
        self.use_eos = use_eos

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward pass.")


class LinearWithAddedEos(BaseAlignmentModule):
    """
    Projects paragraph embeddings with a linear layer and appends a learnable EOS token.
    """
    def __init__(self, input_dim, output_dim, prompt_length, use_eos=True):
        super().__init__(input_dim, output_dim, prompt_length, use_eos)
        self.projection = nn.Linear(input_dim, prompt_length * output_dim)
        if use_eos:
            self.eos_token = nn.Parameter(torch.randn(output_dim))

    def forward(self, paragraphs):
        proj = self.projection(paragraphs)  # [batch_size, prompt_length * output_dim]
        batch_size = paragraphs.size(0)
        prompts = proj.view(batch_size, self.prompt_length, self.output_dim)
        if self.use_eos:
            eos = self.eos_token.unsqueeze(0).expand(batch_size, 1, self.output_dim)
            prompts = torch.cat([prompts, eos], dim=1)
        return prompts


class FFNWithAddedEos(BaseAlignmentModule):
    """
    Uses a small Feed-Forward Network to map paragraph embeddings, then appends an EOS token.
    """
    def __init__(self, input_dim, output_dim, prompt_length, use_eos=True, hidden_dim=512):
        super().__init__(input_dim, output_dim, prompt_length, use_eos)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_length * output_dim),
        )
        if use_eos:
            self.eos_token = nn.Parameter(torch.randn(output_dim))

    def forward(self, paragraphs):
        batch_size = paragraphs.size(0)
        out = self.ffn(paragraphs)
        prompts = out.view(batch_size, self.prompt_length, self.output_dim)
        if self.use_eos:
            eos = self.eos_token.unsqueeze(0).expand(batch_size, 1, self.output_dim)
            prompts = torch.cat([prompts, eos], dim=1)
        return prompts


class PerceiverResampler(BaseAlignmentModule):
    """
    A simplified Perceiver resampler that uses learnable queries and attention.
    """
    def __init__(self, input_dim, output_dim, prompt_length, use_eos=True, num_latents=16):
        super().__init__(input_dim, output_dim, prompt_length, use_eos)
        self.num_latents = num_latents 
        self.latent_queries = nn.Parameter(torch.randn(num_latents, input_dim))
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        self.output_projection = nn.Linear(input_dim, prompt_length * output_dim)
        if use_eos:
            self.eos_token = nn.Parameter(torch.randn(output_dim))

    def forward(self, paragraphs):
        para_batch = paragraphs.unsqueeze(1)
        batch_size = para_batch.size(0)
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, self.num_latents, self.input_dim)
        attended, _ = self.attn(latents, para_batch, para_batch)
        out = self.output_projection(attended.mean(dim=1))
        prompts = out.view(batch_size, self.prompt_length, self.output_dim)
        if self.use_eos:
            eos = self.eos_token.unsqueeze(0).expand(batch_size, 1, self.output_dim)
            prompts = torch.cat([prompts, eos], dim=1)
        return prompts


def build_alignment_module(module_type, input_dim, output_dim, prompt_length, use_eos=True):
    module_type = module_type.lower()
    if module_type == "linearwitheddeos":
        return LinearWithAddedEos(input_dim, output_dim, prompt_length, use_eos)
    elif module_type == "ffnwithaddedos":
        return FFNWithAddedEos(input_dim, output_dim, prompt_length, use_eos)
    elif module_type == "perceiverresampler":
        return PerceiverResampler(input_dim, output_dim, prompt_length, use_eos)
    else:
        raise ValueError(f"Unknown alignment module type: {module_type}")


###############################################
# Modular LangBridge Model
###############################################
class LangBridgeModular(nn.Module):
    """
    A modular LangBridge model that can operate in two modes:
      - If use_paragraph_mode is True, the input paragraph is split into sentences.
        A dynamic aggregator (here, the DynamicRotaryAggregator) combines an arbitrary number
        of sentence embeddings into a single paragraph embedding.
      - If False, the entire paragraph is tokenized as one.
    In both cases, the resulting paragraph embedding is fed into an alignment module
    (which is selected via build_alignment_module) and then passed to the decoder.
    """
    def __init__(self,
                 encoder_model,
                 decoder_model,
                 tokenizer,
                 use_paragraph_mode=True,
                 aggregator_type="dynamic_rotary",  # Options: "dynamic_rotary" or "max_pool"
                 alignment_type="LinearWithAddedEos",
                 fine_tune_encoder=True,
                 max_sentence_length=32,
                 prompt_length=10,
                 rotary_rotary_dim=None):
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.fine_tune_encoder = fine_tune_encoder
        self.max_sentence_length = max_sentence_length
        self.prompt_length = prompt_length
        self.use_paragraph_mode = use_paragraph_mode

        # Freeze decoder parameters.
        for param in self.decoder.parameters():
            param.requires_grad = False
        # Optionally freeze encoder.
        if not self.fine_tune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder_hidden_size = self.encoder.config.hidden_size
        if hasattr(self.decoder.config, 'hidden_size'):
            self.decoder_hidden_size = self.decoder.config.hidden_size
        elif hasattr(self.decoder.config, 'd_model'):
            self.decoder_hidden_size = self.decoder.config.d_model
        else:
            raise ValueError("Decoder config must have either 'hidden_size' or 'd_model'.")

        if self.use_paragraph_mode:
            # Split paragraphs into sentences.
            # Use one aggregator to combine variable number of sentence embeddings.
            if aggregator_type.lower() == "dynamic_rotary":
                # DynamicRotaryAggregator does not require a fixed maximum; it simply accepts N sentences.
                self.aggregator = DynamicRotaryAggregator(self.encoder_hidden_size, rotary_dim=rotary_rotary_dim)
            elif aggregator_type.lower() == "max_pool":
                self.aggregator = MaxPoolAggregator()
            else:
                raise ValueError(f"Unknown aggregator type: {aggregator_type}")
            self.aggregated_dim = self.encoder_hidden_size
        else:
            self.aggregated_dim = self.encoder_hidden_size

        # Build the alignment module using the provided helper.
        self.aligner = build_alignment_module(
            module_type=alignment_type,
            input_dim=self.aggregated_dim,
            output_dim=self.decoder_hidden_size,
            prompt_length=self.prompt_length,
            use_eos=True
        )

    def forward(self, paragraphs, decoder_kwargs=None):
        if decoder_kwargs is None:
            decoder_kwargs = {}
        device = next(self.encoder.parameters()).device

        if self.use_paragraph_mode:
            # 1. Split paragraphs into sentences.
            paragraphs_sentences = [sent_tokenize(p) for p in paragraphs]
            num_sentences = [len(sents) for sents in paragraphs_sentences]
            all_sentences = [sent for sents in paragraphs_sentences for sent in sents]
            if len(all_sentences) == 0:
                raise ValueError("No sentences found in input paragraphs.")
            # 2. Tokenize all sentences.
            encoded = self.tokenizer(
                all_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_sentence_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # 3. Encode sentences and mean-pool token embeddings.
            encoder_outputs = self.encoder(**encoded)
            last_hidden_state = encoder_outputs.last_hidden_state  # [total_sentences, seq_len, hidden_size]
            attention_mask = encoded["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True).float(), min=1e-9)
            sentence_embeddings = sum_embeddings / lengths  # [total_sentences, hidden_size]
            # 4. Reassemble sentence embeddings per paragraph.
            sentence_embeddings_list = []
            idx = 0
            for n in num_sentences:
                if n > 0:
                    sent_emb = sentence_embeddings[idx:idx+n]
                else:
                    sent_emb = torch.zeros(1, self.encoder_hidden_size, device=device)
                sentence_embeddings_list.append(sent_emb)
                idx += n
            # 5. Use the aggregator (dynamic rotary or max-pool) to combine sentence embeddings.
            # For each paragraph, the aggregator returns a single vector of shape [hidden_size].
            paragraph_embeddings = []
            for emb in sentence_embeddings_list:
                paragraph_embeddings.append(self.aggregator(emb))
            paragraph_embeddings = torch.stack(paragraph_embeddings, dim=0)  # [batch_size, hidden_size]
        else:
            # When paragraph mode is off: tokenize full paragraphs.
            encoded = self.tokenizer(
                paragraphs,
                padding=True,
                truncation=True,
                max_length=self.max_sentence_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            encoder_outputs = self.encoder(**encoded)
            last_hidden_state = encoder_outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True).float(), min=1e-9)
            paragraph_embeddings = sum_embeddings / lengths  # [batch_size, hidden_size]

        # 6. Align the aggregated paragraph embeddings to produce the soft prompt.
        soft_prompt = self.aligner(paragraph_embeddings)
        # 7. Feed the soft prompt to the decoder.
        decoder_outputs = self.decoder(inputs_embeds=soft_prompt, **decoder_kwargs)
        return decoder_outputs, soft_prompt
