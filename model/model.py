import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)


class MaxPoolAggregator(nn.Module):
    """
    Collects sentence embeddings and applies max pooling to produce a paragraph embedding.
    """
    def __init__(self):
        super().__init__()

    def forward(self, sentence_embeddings_list, num_sentences, encoder_hidden_size):
        # Pad sequences to a common length
        padded_embeddings = pad_sequence(sentence_embeddings_list, batch_first=True)
        batch_size, _, _ = padded_embeddings.shape
        aggregated_dim = encoder_hidden_size

        # Apply max pooling over valid sentences
        aggregated = []
        for i in range(batch_size):
            valid_outputs = padded_embeddings[i, :num_sentences[i], :]
            if valid_outputs.size(0) > 0:
                pooled = valid_outputs.max(dim=0)[0]
            else:
                pooled = torch.zeros(aggregated_dim, device=valid_outputs.device)
            aggregated.append(pooled)
        return torch.stack(aggregated, dim=0)


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
        # [batch_size, input_dim] -> [batch_size, prompt_length * output_dim]
        proj = self.projection(paragraphs)
        batch_size = paragraphs.size(0)
        prompts = proj.view(batch_size, self.prompt_length, self.output_dim)

        # Append EOS if requested
        if self.use_eos:
            eos = self.eos_token.unsqueeze(0).expand(batch_size, 1, self.output_dim)
            prompts = torch.cat([prompts, eos], dim=1)
        return prompts


class FFNWithAddedEos(BaseAlignmentModule):
    """
    Uses a small Feed-Forward Network to map paragraph embeddings, then appends an EOS token.
    Allows for more complex transformation than a single linear layer.
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
        out = self.ffn(paragraphs)  # [batch_size, prompt_length * output_dim]
        prompts = out.view(batch_size, self.prompt_length, self.output_dim)

        if self.use_eos:
            eos = self.eos_token.unsqueeze(0).expand(batch_size, 1, self.output_dim)
            prompts = torch.cat([prompts, eos], dim=1)
        return prompts


class PerceiverResampler(BaseAlignmentModule):
    """
    A simplified form of Perceiver resampler: uses a set of learnable queries
    that attend to the paragraph embeddings. An EOS token can be added.
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
        # paragraphs: [batch_size, input_dim]
        # Expand paragraph embeddings to shape [batch_size, 1, input_dim]
        para_batch = paragraphs.unsqueeze(1)

        # Repeat latent queries for the entire batch
        batch_size = para_batch.size(0)
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, self.num_latents, self.input_dim)

        # Cross-attend from latents to paragraph embeddings
        # treat paragraphs as "memory", latents as "queries"
        attended, _ = self.attn(latents, para_batch, para_batch)  # [batch_size, num_latents, input_dim]

        # Project to desired dimension and shape
        out = self.output_projection(attended.mean(dim=1))  # [batch_size, prompt_length * output_dim]
        prompts = out.view(batch_size, self.prompt_length, self.output_dim)

        # Optionally append EOS
        if self.use_eos:
            eos = self.eos_token.unsqueeze(0).expand(batch_size, 1, self.output_dim)
            prompts = torch.cat([prompts, eos], dim=1)
        return prompts


def build_alignment_module(module_type, input_dim, output_dim, prompt_length, use_eos=True):
    """
    Constructs the specified alignment module.
    """
    module_type = module_type.lower()
    if module_type == "linearwitheddeos":
        return LinearWithAddedEos(input_dim, output_dim, prompt_length, use_eos)
    elif module_type == "ffnwithaddedos":
        return FFNWithAddedEos(input_dim, output_dim, prompt_length, use_eos)
    elif module_type == "perceiverresampler":
        return PerceiverResampler(input_dim, output_dim, prompt_length, use_eos)
    else:
        raise ValueError(f"Unknown alignment module type: {module_type}")


class LangBridgeModular(nn.Module):
    """
    A PyTorch module that allows modular selection of:
      - Sentence embedding aggregator (e.g., MaxPool or external LSTM aggregator).
      - Alignment module (e.g., LinearWithAddedEos, FFNWithAddedEos, PerceiverResampler).
    """
    def __init__(self,
                 encoder_model,
                 decoder_model,
                 tokenizer,
                 aggregator_type="max_pool",
                 alignment_type="LinearWithAddedEos",
                 fine_tune_encoder=True,
                 max_sentence_length=32,
                 prompt_length=10):
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.fine_tune_encoder = fine_tune_encoder
        self.max_sentence_length = max_sentence_length
        self.prompt_length = prompt_length
        self.aggregator_type = aggregator_type
        self.alignment_type = alignment_type

        # Freeze decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Optionally freeze encoder parameters
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

        # Build aggregator
        if aggregator_type.lower() == "max_pool":
            self.aggregator = MaxPoolAggregator()
            self.aggregated_dim = self.encoder_hidden_size
        elif aggregator_type.lower() == "lstm":
            # For LSTM aggregator, see separate file for implementation details.
            # You could import and instantiate here, e.g.: self.aggregator = LSTMAggregator(...)
            raise NotImplementedError("LSTM aggregator is tracked in a separate file.")
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")

        # Build alignment module
        self.aligner = build_alignment_module(
            module_type=self.alignment_type,
            input_dim=self.aggregated_dim,
            output_dim=self.decoder_hidden_size,
            prompt_length=self.prompt_length,
            use_eos=True
        )

    def forward(self, paragraphs, decoder_kwargs=None):
        if decoder_kwargs is None:
            decoder_kwargs = {}

        batch_size = len(paragraphs)
        device = next(self.encoder.parameters()).device

        # 1. Split paragraphs into sentences and flatten
        paragraphs_sentences = [sent_tokenize(p) for p in paragraphs]
        num_sentences = [len(sents) for sents in paragraphs_sentences]
        all_sentences = [sent for sents in paragraphs_sentences for sent in sents]
        if len(all_sentences) == 0:
            raise ValueError("No sentences found in input paragraphs.")

        # 2. Tokenize
        encoded = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_sentence_length,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # 3. Encode sentences; do mean pooling over tokens
        encoder_outputs = self.encoder(**encoded)
        last_hidden_state = encoder_outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True).float(), min=1e-9)
        sentence_embeddings = sum_embeddings / lengths

        # 4. Reassemble by paragraph
        sentence_embeddings_list = []
        idx = 0
        for n in num_sentences:
            if n > 0:
                sent_emb = sentence_embeddings[idx:idx + n]
            else:
                sent_emb = torch.zeros(1, self.encoder_hidden_size, device=device)
            sentence_embeddings_list.append(sent_emb)
            idx += n

        # 5. Aggregate
        aggregated = self.aggregator(sentence_embeddings_list, num_sentences, self.encoder_hidden_size)

        # 6. Align (project) into soft prompts (or similar)
        soft_prompt = self.aligner(aggregated)

        # 7. Decode
        decoder_outputs = self.decoder(inputs_embeds=soft_prompt, **decoder_kwargs)

        return decoder_outputs, soft_prompt