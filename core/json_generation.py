"""JSON token constraints using LM Format Enforcer's public, framework-free API.

The released Transformers integration imports a tokenizer location removed in
Transformers 5. This small adapter avoids that import and private generate hooks.
The tokenizer index is reused; every generation gets a fresh schema parser.
"""

from copy import deepcopy


class JsonGenerationConstraint:
    def __init__(self, tokenizer, eos_token_ids=None):
        try:
            import lmformatenforcer
        except ImportError as error:
            raise RuntimeError(
                "Patient JSON generation requires lm-format-enforcer==0.11.3. "
                "Install it in the active environment: pip install lm-format-enforcer==0.11.3"
            ) from error
        self._library = lmformatenforcer
        self.tokenizer = tokenizer
        eos = tokenizer.eos_token_id if eos_token_ids is None else eos_token_ids
        if isinstance(eos, int):
            eos = [eos]
        if not isinstance(eos, list) or not eos or any(type(token) is not int for token in eos):
            raise ValueError("JSON generation requires valid model EOS token IDs.")

        # Contextual decoding preserves leading spaces for tokenizers whose
        # single-token decoding treats word beginnings differently.
        anchor = tokenizer.encode("0", add_special_tokens=False)
        anchor_text = tokenizer.decode(anchor, clean_up_tokenization_spaces=False)
        special = set(tokenizer.all_special_ids) | set(eos)
        regular = []
        for token_id in range(len(tokenizer)):
            if token_id in special:
                continue
            after_anchor = tokenizer.decode(anchor + [token_id], clean_up_tokenization_spaces=False)
            if not after_anchor.startswith(anchor_text):
                raise ValueError("Tokenizer contextual decoding is incompatible with JSON constraints.")
            piece = after_anchor[len(anchor_text):]
            isolated = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if piece:
                regular.append((token_id, piece, len(piece) > len(isolated)))
        self.tokenizer_data = lmformatenforcer.TokenEnforcerTokenizerData(
            regular, self._decode, eos, use_bitmask=False, vocab_size=len(tokenizer))

    def _decode(self, tokens):
        # A byte-token sequence may end midway through a Unicode character.
        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False).rstrip("\ufffd")

    def for_schema(self, schema):
        if not isinstance(schema, dict):
            raise ValueError("json_schema must be an object.")
        parser = self._library.JsonSchemaParser(deepcopy(schema))
        enforcer = self._library.TokenEnforcer(self.tokenizer_data, parser)
        # Required-field order is otherwise only a schema listing, not a
        # generation order. TokenEnforcer replaces parser.config at creation.
        # State must precede speech so it can condition this same output call.
        parser.config.force_json_field_order = True

        def allowed_tokens(batch_id, input_ids):
            return enforcer.get_allowed_tokens(input_ids.tolist()).allowed_tokens

        return allowed_tokens
