# model.py
"""
DDMClassifier updated to implement insertion-style masking (CLS 后插入 M 个 [MASK])
and to set position-embeddings of mask tokens to zero (approximation via inputs_embeds).

Requirements: local pretrained bert at Config.model_name_or_path (offline).
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from config import Config
import random
import math


class DDMClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        # load bert locally
        self.bert = BertModel.from_pretrained(Config.model_name_or_path, local_files_only=True)
        hidden_size = self.bert.config.hidden_size

        # classification head
        self.dropout = nn.Dropout(Config.dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(Config.model_name_or_path, local_files_only=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                labels: Optional[torch.LongTensor] = None,
                apply_train_mask: bool = False,
                train_mask_ratio: float = None,
                insert_mode: bool = True,
                freq_dict: Optional[Dict[int, int]] = None,
                apply_inference_mask: bool = False,
                suspicious_ratio: float = None):
        """
        If apply_train_mask and insert_mode -> use insertion-style masking during training.
        If apply_inference_mask -> perform inference-time masking (insertion-style) using freq_dict or batch fallback.
        """
        # optionally apply training-time insertion mask
        if apply_train_mask:
            ratio = train_mask_ratio if train_mask_ratio is not None else Config.mask_ratio
            if insert_mode:
                input_ids, attention_mask = self.train_mask_inject_insert(input_ids, attention_mask, ratio)
            else:
                input_ids = self.train_mask_inject_replace(input_ids, attention_mask, ratio)

        # optionally apply inference-time masking (insert mode)
        if apply_inference_mask:
            sus_ratio = suspicious_ratio if suspicious_ratio is not None else Config.suspicious_ratio
            input_ids, attention_mask = self.mask_suspicious_tokens_insert(input_ids, attention_mask,
                                                                          suspicious_ratio=sus_ratio,
                                                                          freq_dict=freq_dict)

        # Instead of passing input_ids directly, we'll build inputs_embeds so we can zero out
        # positional embeddings for mask tokens (as paper suggests).
        inputs_embeds, new_attention_mask = self._build_inputs_embeds_with_masked_pos_zero(input_ids, attention_mask)

        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=new_attention_mask, return_dict=True)
        pooled = outputs.pooler_output  # (B, H)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}

    # -------------------------
    # Training-time insertion
    # -------------------------
    def train_mask_inject_insert(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, mask_ratio: float):
        """
        Insert M = floor(mask_ratio * valid_len) [MASK] tokens immediately after CLS (index 0).
        Shift original tokens to the right; truncate trailing tokens to keep length unchanged.
        Update attention_mask accordingly.
        Returns (new_input_ids, new_attention_mask)
        """
        batch_size, seq_len = input_ids.size()
        new_input_ids = input_ids.clone()
        new_attention_mask = attention_mask.clone()

        for b in range(batch_size):
            # collect valid positions (exclude cls, sep, pad)
            valid_positions = []
            for i in range(1, seq_len):  # start after CLS
                if attention_mask[b, i].item() == 0:
                    continue
                if input_ids[b, i].item() == self.sep_token_id:
                    # include sep position as valid token but careful with shifting
                    valid_positions.append(i)
                    break
                valid_positions.append(i)
            valid_len = len(valid_positions)
            if valid_len == 0:
                continue
            m = max(1, int(math.floor(mask_ratio * valid_len)))
            # perform insertion after CLS: we'll create a new sequence by:
            # [CLS] [MASK]*m orig[1:seq_len-m] (i.e., shift right and drop last m tokens)
            # But must ensure SEP remains at the end if possible; we'll do a simple safe shift:
            # Build list of token ids for this sequence:
            orig = list(input_ids[b].cpu().tolist())
            orig_attn = list(attention_mask[b].cpu().tolist())
            # Build token list excluding trailing pads for clarity
            # We'll attempt to insert m masks after index 0
            masks = [self.mask_token_id] * m
            # New sequence: CLS + masks + orig[1:seq_len - m]
            right_take = orig[1: seq_len - m] if seq_len - m > 1 else []
            new_seq = [orig[0]] + masks + right_take
            # If new_seq shorter than seq_len, append remaining tokens from orig (may include sep)
            if len(new_seq) < seq_len:
                # append the remaining tokens from orig that were truncated earlier (to try keep sep)
                # We'll fill from the tail of orig up to seq_len
                tail_needed = seq_len - len(new_seq)
                # take from the tail of orig that are not already included
                tail = orig[seq_len - tail_needed: seq_len]
                new_seq += tail
            # Ensure length
            new_seq = new_seq[:seq_len]
            # Attention: recompute attention mask: positions that are pad in orig should become whatever tokens we set
            new_attn = [1 if tid != self.pad_token_id else 0 for tid in new_seq]
            new_input_ids[b] = torch.tensor(new_seq, dtype=input_ids.dtype)
            new_attention_mask[b] = torch.tensor(new_attn, dtype=attention_mask.dtype)

        return new_input_ids.to(input_ids.device), new_attention_mask.to(attention_mask.device)

    def train_mask_inject_replace(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, mask_ratio: float):
        """
        Fallback: random replace (original implementation).
        """
        input_ids = input_ids.clone()
        batch_size, seq_len = input_ids.size()
        special_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        for b in range(batch_size):
            valid_positions = []
            for i in range(seq_len):
                tid = int(input_ids[b, i].item())
                if tid in special_ids:
                    continue
                if attention_mask[b, i].item() == 0:
                    continue
                valid_positions.append(i)
            if len(valid_positions) == 0:
                continue
            m_count = max(1, int(len(valid_positions) * mask_ratio))
            chosen = random.sample(valid_positions, min(m_count, len(valid_positions)))
            for pos in chosen:
                input_ids[b, pos] = self.mask_token_id
        return input_ids

    # -------------------------
    # Inference-time insertion masking
    # -------------------------
    def mask_suspicious_tokens_insert(self,
                                  input_ids: torch.LongTensor,
                                  attention_mask: torch.LongTensor,
                                  suspicious_ratio: float = None,
                                  freq_dict: Optional[Dict[int, int]] = None):
        """
        修改版：
        - 不再插入 mask。
        - 直接将 top_pos 上的可疑 token 替换成 [MASK]。
        """
        suspicious_ratio = suspicious_ratio if suspicious_ratio is not None else Config.suspicious_ratio
        batch_size, seq_len = input_ids.size()
        new_input_ids = input_ids.clone()
        new_attention_mask = attention_mask.clone()

        # 计算频率字典（batch 内或外部传入）
        if freq_dict is None:
            flat = input_ids.view(-1).tolist()
            freq = {}
            for tid in flat:
                freq[tid] = freq.get(tid, 0) + 1
        else:
            freq = {int(k): int(v) for k, v in freq_dict.items()}

        for b in range(batch_size):
            # 收集候选 token
            candidates = []
            for i in range(seq_len):
                tid = int(input_ids[b, i].item())
                if tid in {self.cls_token_id, self.sep_token_id, self.pad_token_id}:
                    continue
                if attention_mask[b, i].item() == 0:
                    continue
                f = freq.get(tid, 0)
                score = 1.0 / (f + 1e-12) if f > 0 else 1e9
                candidates.append((i, score))

            if len(candidates) == 0:
                continue

            # 选出 top-K 可疑 token
            candidates.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(len(candidates) * suspicious_ratio))
            top_pos = [pos for pos, _ in candidates[:k]]

            # 直接将这些 token 替换成 [MASK]
            for p in top_pos:
                new_input_ids[b, p] = self.mask_token_id

            text_after_mask = self.tokenizer.decode(
                new_input_ids[b][attention_mask[b] == 1].tolist(),
                skip_special_tokens=False
            )
            # print(f"[Sample {b}] After masking:\n{text_after_mask}\n")
            with open("sample_after_mask.txt", "a", encoding="utf-8") as f:
                f.write(f"[Sample {b}] After masking:\n{text_after_mask}\n\n")
            # attention mask 不需要改，仍然沿用原 attention_mask
            # 因为 [MASK] 应该被正常注意到

        return new_input_ids.to(input_ids.device), new_attention_mask.to(attention_mask.device)


    # -------------------------
    # Build inputs_embeds and zero positional embeddings for mask tokens
    # -------------------------
    def _build_inputs_embeds_with_masked_pos_zero(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        Construct inputs_embeds = word_emb + pos_emb + token_type_emb, but zero out pos_emb rows
        where input_ids == mask_token_id (so mask tokens do not carry positional info).
        Return (inputs_embeds, attention_mask) ready to feed into bert(..., inputs_embeds=..., attention_mask=...)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        # word embeddings: (B, L, H)
        word_emb = self.bert.embeddings.word_embeddings(input_ids.to(self.bert.device))
        # position ids: normally 0..seq_len-1
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.bert.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.bert.embeddings.position_embeddings(position_ids)
        # token_type ids: default all zeros
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=self.bert.device)
        token_type_emb = self.bert.embeddings.token_type_embeddings(token_type_ids)

        # Zero out position embeddings where input_ids == mask_token_id
        mask_positions = (input_ids == self.mask_token_id).to(self.bert.device)  # (B, L)
        # expand mask to (B, L, H)
        mask_positions_exp = mask_positions.unsqueeze(-1).expand(-1, -1, pos_emb.size(-1))
        pos_emb = pos_emb.masked_fill(mask_positions_exp, 0.0)

        inputs_embeds = word_emb + pos_emb + token_type_emb
        # return inputs_embeds on bert device and attention_mask as is (may need to move device)
        return inputs_embeds.to(self.bert.device), attention_mask.to(self.bert.device)

    # -------------------------
    # Save / Load helpers
    # -------------------------
    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        self.tokenizer.save_pretrained(path)
        self.bert.config.save_pretrained(path)
        print(f"Model saved to {path}")

    def load(self, path: str, map_location=None):
        self.bert = BertModel.from_pretrained(path, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        state = torch.load(f"{path}/pytorch_model.bin", map_location=map_location)
        self.load_state_dict(state)
        print(f"Model loaded from {path}")
