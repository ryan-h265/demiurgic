"""
Demiurgic GPT Model.

Main model implementation combining all components into a complete
decoder-only transformer for code generation.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Optional, Tuple, Union

from .config import DemiurgicConfig
from .transformer import DemiurgicDecoderLayer
from .normalization import RMSNorm


class DemiurgicPreTrainedModel(nn.Module):
    """
    Base class for Demiurgic models.

    Handles weight initialization and provides utility methods.
    """

    config_class = DemiurgicConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DemiurgicDecoderLayer"]

    def __init__(self, config: DemiurgicConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        """Initialize weights following GPT-style initialization."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get number of parameters in the model.

        Args:
            only_trainable: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class DemiurgicModel(DemiurgicPreTrainedModel):
    """
    Demiurgic transformer model (base model without language modeling head).

    This is the core transformer that outputs hidden states.
    For language modeling, use DemiurgicForCausalLM.

    Args:
        config: Model configuration
    """

    def __init__(self, config: DemiurgicConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.embedding_dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [DemiurgicDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gradient checkpointing
        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights and apply final processing."""
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for this model."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for this model."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, torch.FloatTensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch_size, seq_len] Token indices
            attention_mask: [batch_size, seq_len] Attention mask
            position_ids: [batch_size, seq_len] Position indices
            past_key_values: Cached key/values for efficient generation
            inputs_embeds: [batch_size, seq_len, hidden_size] Pre-computed embeddings
            use_cache: Whether to return key/values for caching
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict (not implemented yet)

        Returns:
            Tuple of (last_hidden_state, past_key_values, hidden_states, attentions)
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Apply embedding dropout
        hidden_states = self.embed_dropout(inputs_embeds)

        # Causal mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing: don't use cache during training
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,  # past_key_value must be None for checkpointing
                    output_attentions,
                    False,  # use_cache must be False for checkpointing
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            # Only access cache if we actually used it (not in gradient checkpointing)
            if use_cache and not (self.gradient_checkpointing and self.training):
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return (hidden_states, next_decoder_cache, all_hidden_states, all_self_attns)

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """
        Create causal attention mask.

        Creates a 4D attention mask from a 2D tensor mask.
        Mask values selected in [0, 1]:
        - 1 for tokens that are NOT MASKED
        - 0 for tokens that are MASKED
        """
        # Create causal mask
        # [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
            expanded_attn_mask = self._expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for autoregressive decoding.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
                dim=-1,
            )
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len].
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class DemiurgicForCausalLM(DemiurgicPreTrainedModel):
    """
    Demiurgic model with language modeling head for autoregressive generation.

    This is the main model for training and inference.

    Args:
        config: Model configuration
    """

    def __init__(self, config: DemiurgicConfig):
        super().__init__(config)
        self.model = DemiurgicModel(config)
        self.vocab_size = config.vocab_size

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights with input embeddings if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights."""
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for this model."""
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for this model."""
        self.model.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory: str):
        """
        Save model and configuration to a directory.

        Args:
            save_directory: Directory to save model to
        """
        import json
        from pathlib import Path

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = save_dir / 'pytorch_model.bin'
        torch.save(self.state_dict(), model_path)

        # Save config
        config_path = save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, **kwargs):
        """
        Load model from a directory.

        Args:
            pretrained_model_path: Path to saved model directory
            **kwargs: Additional arguments (e.g., device)

        Returns:
            Loaded model
        """
        import json
        from pathlib import Path

        model_dir = Path(pretrained_model_path)

        # Load config
        config_path = model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Remove model_type if present
        config_dict.pop('model_type', None)

        # Create model
        config = DemiurgicConfig(**config_dict)
        model = cls(config)

        # Load weights
        model_path = model_dir / 'pytorch_model.bin'
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

        # Move to device if specified
        device = kwargs.get('device', 'cpu')
        model = model.to(device)

        print(f"Model loaded from {pretrained_model_path}")
        return model

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """
        Resize input and output token embeddings to new vocabulary size.

        Args:
            new_num_tokens: New vocabulary size

        Returns:
            The new input embeddings
        """
        old_num_tokens = self.config.vocab_size

        if new_num_tokens == old_num_tokens:
            return self.get_input_embeddings()

        # Resize input embeddings
        old_embeddings = self.get_input_embeddings()
        new_embeddings = nn.Embedding(
            new_num_tokens,
            self.config.hidden_size,
            padding_idx=self.config.pad_token_id,
        )

        # Initialize new embeddings
        new_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        # Copy old embeddings
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        self.set_input_embeddings(new_embeddings)

        # Resize output embeddings (lm_head)
        old_lm_head = self.get_output_embeddings()
        new_lm_head = nn.Linear(
            self.config.hidden_size,
            new_num_tokens,
            bias=False,
        )

        # Initialize new output embeddings
        new_lm_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        # Copy old output embeddings
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]

        # Handle tied embeddings
        if self.config.tie_word_embeddings:
            new_lm_head.weight = new_embeddings.weight

        self.set_output_embeddings(new_lm_head)

        # Update config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens

        return self.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, torch.FloatTensor]:
        """
        Forward pass with language modeling.

        Args:
            input_ids: [batch_size, seq_len] Token indices
            attention_mask: [batch_size, seq_len] Attention mask
            position_ids: [batch_size, seq_len] Position indices
            past_key_values: Cached key/values
            inputs_embeds: Pre-computed embeddings
            labels: [batch_size, seq_len] Labels for computing loss
            use_cache: Whether to cache key/values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return dict

        Returns:
            Tuple of (loss, logits, past_key_values, hidden_states, attentions)
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # Compute logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return (loss, logits) + outputs[1:]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """
        Simple greedy/sampling generation.

        For production, use HuggingFace's generate() method.
        This is a minimal implementation for testing.

        Args:
            input_ids: [batch_size, seq_len] Starting tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample (vs greedy)

        Returns:
            Generated token IDs
        """
        for _ in range(max_length):
            # Forward pass
            outputs = self.forward(input_ids, use_cache=False)
            logits = outputs[1]  # [batch_size, seq_len, vocab_size]

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Sampling
            if do_sample:
                # Top-k
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Top-p (nucleus)
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if EOS
            if (next_token == self.config.eos_token_id).all():
                break

        return input_ids
