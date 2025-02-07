import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import nltk
from nltk.tokenize import sent_tokenize

# Ensure the sentence splitter is available.
nltk.download('punkt', quiet=True)


class LangBridgeWithLSTM(nn.Module):
    """
    A PyTorch module that implements LangBridge's architecture with an LSTM aggregator.
    
    It expects:
      - A pretrained encoder model (from HuggingFace Transformers)
      - A pretrained decoder model (also from Transformers; assumed to be frozen)
      - A tokenizer (for the encoder) to tokenize individual sentences.
      
    The forward pass:
      1. Receives a batch of raw paragraph strings.
      2. Splits each paragraph into sentences (using nltk.sent_tokenize).
      3. Tokenizes each sentence (with padding/truncation to a max length).
      4. Encodes all sentences via the encoder model.
      5. Re-assembles the sentence embeddings into a padded batch,
         packs them for an LSTM aggregator (bidirectional with dropout 0.1 and a number of layers based on config).
      6. Mean-pools the LSTM outputs over the sentence dimension to form a single paragraph embedding.
      7. Projects this aggregated vector to a “soft prompt” sequence (of fixed prompt_length) matching the decoder's input dimension.
      8. Feeds the soft prompt into the frozen decoder.
      
    The training objective is expected to align soft prompts from parallel texts.
    """
    def __init__(self,
                 encoder_model,
                 decoder_model,
                 tokenizer,
                 fine_tune_encoder=True,
                 lstm_num_layers=1,
                 max_sentence_length=32,
                 prompt_length=10):
        """
        Args:
            encoder_model: Pretrained encoder (e.g., from AutoModel.from_pretrained).
            decoder_model: Pretrained decoder (e.g., T5 or GPT variant) - it will be frozen.
            tokenizer: The tokenizer corresponding to the encoder model.
            fine_tune_encoder (bool): Whether to update the encoder during training.
            lstm_num_layers (int): Number of layers for the LSTM aggregator.
            max_sentence_length (int): Maximum number of tokens for each sentence.
            prompt_length (int): Number of soft prompt tokens to produce for the decoder.
        """
        super(LangBridgeWithLSTM, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.fine_tune_encoder = fine_tune_encoder
        self.max_sentence_length = max_sentence_length
        self.prompt_length = prompt_length

        # Freeze decoder parameters (always frozen as in LangBridge)
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Optionally freeze encoder parameters
        if not self.fine_tune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Determine the encoder’s hidden size (e.g., from BERT, RoBERTa, etc.)
        self.encoder_hidden_size = self.encoder.config.hidden_size

        # LSTM aggregator: Input is a sentence embedding of size encoder_hidden_size.
        # Set dropout=0.1 (if more than 1 layer) and bidirectional=True.
        self.lstm = nn.LSTM(
            input_size=self.encoder_hidden_size,
            hidden_size=self.encoder_hidden_size,
            num_layers=lstm_num_layers,
            dropout=0.1 if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        # Because the LSTM is bidirectional, its output dimension is 2*encoder_hidden_size.
        self.aggregated_dim = 2 * self.encoder_hidden_size

        # Determine the decoder’s expected hidden size.
        if hasattr(self.decoder.config, 'hidden_size'):
            self.decoder_hidden_size = self.decoder.config.hidden_size
        elif hasattr(self.decoder.config, 'd_model'):
            self.decoder_hidden_size = self.decoder.config.d_model
        else:
            raise ValueError("Decoder config must have either 'hidden_size' or 'd_model'.")

        # Bridge layer: project the aggregated vector to a fixed-length soft prompt.
        # The soft prompt will be a sequence of prompt_length embeddings (each of size decoder_hidden_size).
        self.bridge = nn.Linear(self.aggregated_dim, prompt_length * self.decoder_hidden_size)

    def forward(self, paragraphs, decoder_kwargs=None):
        """
        Forward pass.
        
        Args:
            paragraphs (List[str]): A list of raw paragraph strings (batch size = len(paragraphs)).
            decoder_kwargs (dict, optional): Additional keyword arguments to pass to the decoder.
            
        Returns:
            A tuple (decoder_outputs, soft_prompt) where:
              - decoder_outputs: The output from the frozen decoder.
              - soft_prompt: The generated soft prompt (of shape [batch_size, prompt_length, decoder_hidden_size]).
        """
        if decoder_kwargs is None:
            decoder_kwargs = {}

        batch_size = len(paragraphs)

        # 1. Split each paragraph into sentences.
        paragraphs_sentences = [sent_tokenize(p) for p in paragraphs]  # List[List[str]]
        num_sentences = [len(sents) for sents in paragraphs_sentences]

        # 2. Flatten all sentences from the batch for efficient parallel tokenization.
        all_sentences = [sent for sents in paragraphs_sentences for sent in sents]
        if len(all_sentences) == 0:
            raise ValueError("No sentences found in the input paragraphs.")

        # 3. Tokenize all sentences (using the provided tokenizer).
        encoded = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_sentence_length,
            return_tensors='pt'
        )
        device = next(self.encoder.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # 4. Encode all sentences.
        encoder_outputs = self.encoder(**encoded)
        # Extract the last hidden state and perform mean pooling over tokens.
        last_hidden_state = encoder_outputs.last_hidden_state  # [num_sentences, seq_len, hidden_size]
        attention_mask = encoded["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True).float(), min=1e-9)
        sentence_embeddings = sum_embeddings / lengths  # [num_sentences, encoder_hidden_size]

        # 5. Reassemble the sentence embeddings back into paragraphs.
        sentence_embeddings_list = []
        index = 0
        for n in num_sentences:
            if n > 0:
                sent_emb = sentence_embeddings[index:index + n]
            else:
                # In case a paragraph has no sentences (should rarely happen), add a zero vector.
                sent_emb = torch.zeros(1, self.encoder_hidden_size, device=device)
            sentence_embeddings_list.append(sent_emb)
            index += n

        # 6. Pad the list of sentence sequences so that they all have the same number of sentences.
        padded_embeddings = pad_sequence(sentence_embeddings_list, batch_first=True)  # [batch_size, max_sentences, encoder_hidden_size]
        lengths_tensor = torch.tensor(num_sentences, dtype=torch.long, device=device)

        # 7. Pack the padded sequences and feed them through the LSTM aggregator.
        packed = pack_padded_sequence(padded_embeddings, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(packed)
        lstm_out_unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [batch_size, max_sentences, 2*encoder_hidden_size]

        # 8. Aggregate LSTM outputs to obtain a single paragraph representation.
        # Here we use mean pooling over the valid time steps (sentences).
        aggregated = []
        for i in range(batch_size):
            valid_outputs = lstm_out_unpacked[i, :num_sentences[i], :]  # [num_sentences_i, 2*encoder_hidden_size]
            if valid_outputs.size(0) > 0:
                pooled = valid_outputs.mean(dim=0)  # [2*encoder_hidden_size]
            else:
                pooled = torch.zeros(self.aggregated_dim, device=device)
            aggregated.append(pooled)
        aggregated = torch.stack(aggregated, dim=0)  # [batch_size, 2*encoder_hidden_size]

        # 9. Project the aggregated representation to a fixed-length soft prompt.
        bridge_output = self.bridge(aggregated)  # [batch_size, prompt_length * decoder_hidden_size]
        soft_prompt = bridge_output.view(batch_size, self.prompt_length, self.decoder_hidden_size)

        # 10. Pass the soft prompt to the frozen decoder.
        # (It is assumed that the decoder accepts an 'inputs_embeds' argument.)
        decoder_outputs = self.decoder(inputs_embeds=soft_prompt, **decoder_kwargs)

        return decoder_outputs, soft_prompt
