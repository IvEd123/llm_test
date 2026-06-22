"""
gpt2_walker.py — пошаговый ручной форвард-пасс GPT-2 с сохранением состояния,
модульными визуализаторами/сейверами и батч-обработкой.

Архитектура (см. README в конце файла docstring'ом не дублируется, см. ноутбук):

  WalkerState   — состояние одного прогона: residual stream, текущий блок,
                  текущая детальная стадия внутри блока, кэш тензоров.
                  Можно "заморозить" на любом месте и продолжить позже —
                  не нужно пересчитывать всё с нуля, чтобы добраться
                  до интересующего этапа.

  GPT2Walker    — движок. Умеет:
                  * start()      — токенизация + эмбеддинги (+ эталонный
                                    прогон HF-модели для сверки)
                  * step()       — ОДИН детальный под-шаг внутри блока
                                    (ln_1, qkv, heads, scores, mask, softmax,
                                    attn_v, attn_proj, resid1, ln_2, mlp_fc,
                                    mlp_act, mlp_proj, resid2)
                  * run_block()  — весь блок разом (detail=False, быстро)
                                    или по шагам (detail=True, подробно)
                  * skip_to()    — быстро доехать до нужного блока/стадии,
                                    не считая всё пошагово с нуля
                  * finalize()   — ln_f + lm_head
                  * decode_topk()— топ-k предсказаний следующего токена
                  * verify()     — max abs diff vs эталонные outputs HF

  Recorder      — базовый класс модульного визуализатора/сейвера. Каждый
                  включается/выключается независимо (recorder.enabled),
                  реагирует только на свои стадии (recorder.stages).

  run_pipeline()— грубый запуск "всех или некоторых" этапов модели целиком
                  (эмбеддинги -> N блоков -> lm_head -> decode).

  run_batch()   — прогон на списке промптов с выгрузкой выбранных тензоров
                  (например, attention 3-го слоя) в файл.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except ImportError:  # pragma: no cover
    _HAS_PLOTTING = False


# Упорядоченный список детальных стадий внутри ОДНОГО блока GPT-2.
BLOCK_STAGES: Tuple[str, ...] = (
    "ln_1", "qkv", "heads", "scores", "mask", "softmax",
    "attn_v", "attn_proj", "resid1",
    "ln_2", "mlp_fc", "mlp_act", "mlp_proj", "resid2",
)


# --------------------------------------------------------------------------
# Состояние
# --------------------------------------------------------------------------

@dataclass
class WalkerState:
    """Состояние одного прогона модели на конкретном промпте.

    Хранит residual stream (`x`), на каком блоке/стадии мы остановились
    и кэш промежуточных тензоров ТЕКУЩЕГО блока (`cache`). Объект можно
    держать в переменной сколько угодно и продолжать с него позже —
    это и есть "сохранение состояния модели" из пункта 2 требований.
    """

    prompt: str
    ids: torch.Tensor
    tokens: List[str]

    layer_idx: int = -1          # -1 = блоки ещё не начаты (только эмбеддинги)
    stage: str = "embed"         # имя последней завершённой стадии
    x: Optional[torch.Tensor] = None   # текущий residual stream

    cache: Dict[str, torch.Tensor] = field(default_factory=dict)  # тензоры текущего блока
    logits: Optional[torch.Tensor] = None

    reference: Optional[Dict[str, Any]] = None  # эталонные outputs HF-модели (для verify())

    def clone(self) -> "WalkerState":
        import copy
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (f"WalkerState(prompt={self.prompt!r}, layer_idx={self.layer_idx}, "
                f"stage={self.stage!r}, seq_len={self.ids.shape[1]})")


# --------------------------------------------------------------------------
# Движок
# --------------------------------------------------------------------------

class GPT2Walker:
    """Движок ручного форвард-пасса GPT-2 (см. docstring модуля)."""

    def __init__(self, model, tokenizer, recorders: Optional[List["Recorder"]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.recorders: List[Recorder] = list(recorders) if recorders else []

        cfg = model.config
        self.embed_dim = getattr(cfg, "hidden_size", None) or cfg.n_embd
        self.num_heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.num_layers = len(model.transformer.h)

    def __repr__(self) -> str:
        recs = ", ".join(f"{r.name}({'on' if r.enabled else 'off'})" for r in self.recorders)
        return (f"GPT2Walker(layers={self.num_layers}, heads={self.num_heads}, "
                f"embed_dim={self.embed_dim}) recorders=[{recs}]")

    # ---------------- управление рекордерами ----------------

    def add_recorder(self, recorder: "Recorder") -> "Recorder":
        self.recorders.append(recorder)
        return recorder

    def remove_recorder(self, name: str) -> None:
        self.recorders = [r for r in self.recorders if r.name != name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        found = False
        for r in self.recorders:
            if r.name == name:
                r.enabled = enabled
                found = True
        if not found:
            raise KeyError(f"Рекордер {name!r} не найден. Есть: "
                            f"{[r.name for r in self.recorders]}")

    def _dispatch(self, stage: str, state: WalkerState, **extra) -> None:
        for r in self.recorders:
            if r.should_fire(stage, state.layer_idx):
                try:
                    r.fire(stage=stage, layer_idx=state.layer_idx, state=state,
                           walker=self, **extra)
                except KeyError as e:
                    print(f"  [{r.name}] пропущен на стадии {stage!r}: нет тензора {e} "
                          f"в кэше (вероятно, использован detail=False)")

    # ---------------- старт ----------------

    def start(self, prompt: str, with_reference: bool = True) -> WalkerState:
        """Токенизация + эмбеддинги (wte + wpe). Опционально считает один
        эталонный прогон HF-модели целиком — он используется только для
        сверки (verify()/ReferenceDiffRecorder), на дальнейшие шаги не влияет.
        """
        tokens = self.tokenizer.tokenize(prompt)
        ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])

        reference = None
        if with_reference:
            with torch.no_grad():
                ref_out = self.model(ids, output_attentions=True,
                                      output_hidden_states=True, return_dict=True)
            reference = {
                "hidden_states": ref_out.hidden_states,
                "attentions": ref_out.attentions,
                "logits": ref_out.logits,
            }

        state = WalkerState(prompt=prompt, ids=ids, tokens=tokens, reference=reference)

        seq_len = ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        with torch.no_grad():
            tok_emb = self.model.transformer.wte(ids)
            pos_emb = self.model.transformer.wpe(position_ids)
        state.x = tok_emb + pos_emb
        state.layer_idx = -1
        state.stage = "embed"

        self._dispatch("embed", state, tok_emb=tok_emb, pos_emb=pos_emb)
        return state

    # ---------------- один детальный шаг ----------------

    def step(self, state: WalkerState) -> WalkerState:
        """Продвигает state ровно на одну детальную стадию (см. BLOCK_STAGES).
        Автоматически переходит на следующий блок, когда текущий завершён
        (stage == 'resid2') или это самый первый вызов после эмбеддингов.
        """
        if state.stage == "lm_head":
            raise StopIteration("Пайплайн уже завершён (lm_head посчитан).")

        if state.stage in ("embed", "resid2"):
            new_layer = state.layer_idx + 1
            if new_layer >= self.num_layers:
                raise StopIteration(
                    f"Все {self.num_layers} блок(ов) пройдены. "
                    f"Вызовите walker.finalize(state) для ln_f + lm_head.")
            state.layer_idx = new_layer
            state.cache = {"x_in": state.x}
            next_stage = "ln_1"
        else:
            idx = BLOCK_STAGES.index(state.stage)
            next_stage = BLOCK_STAGES[idx + 1]

        self._compute_stage(state, next_stage)
        state.stage = next_stage
        self._dispatch(next_stage, state)
        return state

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = t.shape[0], t.shape[1]
        t = t.view(bsz, seq_len, self.num_heads, self.head_dim)
        return t.permute(0, 2, 1, 3)

    @staticmethod
    def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def _compute_stage(self, state: WalkerState, stage: str) -> None:
        block = self.model.transformer.h[state.layer_idx]
        c = state.cache
        with torch.no_grad():
            if stage == "ln_1":
                c["ln1_out"] = block.ln_1(c["x_in"])

            elif stage == "qkv":
                qkv = block.attn.c_attn(c["ln1_out"])
                q, k, v = qkv.split(self.embed_dim, dim=2)
                c["q"], c["k"], c["v"] = q, k, v

            elif stage == "heads":
                c["q_h"] = self._split_heads(c["q"])
                c["k_h"] = self._split_heads(c["k"])
                c["v_h"] = self._split_heads(c["v"])

            elif stage == "scores":
                scaling = getattr(block.attn, "scaling", None) or (1.0 / math.sqrt(self.head_dim))
                c["scores"] = torch.matmul(c["q_h"], c["k_h"].transpose(-1, -2)) * scaling

            elif stage == "mask":
                seq_len = state.ids.shape[1]
                c["scores_masked"] = c["scores"] + self._causal_mask(seq_len, c["scores"].dtype)

            elif stage == "softmax":
                c["attn_weights"] = F.softmax(c["scores_masked"], dim=-1)

            elif stage == "attn_v":
                c["attn_out_heads"] = torch.matmul(c["attn_weights"], c["v_h"])

            elif stage == "attn_proj":
                bsz, seq_len = state.ids.shape
                merged = c["attn_out_heads"].transpose(1, 2).contiguous()
                merged = merged.reshape(bsz, seq_len, self.embed_dim)
                c["attn_out"] = block.attn.c_proj(merged)

            elif stage == "resid1":
                c["resid1"] = c["x_in"] + c["attn_out"]

            elif stage == "ln_2":
                c["ln2_out"] = block.ln_2(c["resid1"])

            elif stage == "mlp_fc":
                c["mlp_pre_act"] = block.mlp.c_fc(c["ln2_out"])

            elif stage == "mlp_act":
                c["mlp_post_act"] = block.mlp.act(c["mlp_pre_act"])

            elif stage == "mlp_proj":
                c["mlp_out"] = block.mlp.c_proj(c["mlp_post_act"])

            elif stage == "resid2":
                c["resid2"] = c["resid1"] + c["mlp_out"]
                state.x = c["resid2"]

            else:
                raise ValueError(f"Неизвестная стадия: {stage!r}")

    # ---------------- блок целиком (быстро/детально) ----------------

    def run_block(self, state: WalkerState, detail: bool = True,
                   until: Optional[str] = None) -> WalkerState:
        """Прогоняет ОДИН (следующий) блок.

        detail=True  — по одной стадии (как в исходном ноутбуке), вызывая
                        рекордеры на каждой. `until` — необязательная
                        конечная стадия (например 'softmax'), если не нужен
                        весь блок целиком.
        detail=False — родной forward блока одним вызовом (быстро).
                        Рекордерам в этом случае доступны только
                        cache['x_in'] и cache['resid2'] — детальные
                        промежуточные тензоры (q/k/v, attn_weights, ...)
                        не считаются и не сохраняются.
        """
        if detail:
            target = until or "resid2"
            if target not in BLOCK_STAGES:
                raise ValueError(f"Неизвестная целевая стадия: {target!r}. "
                                  f"Допустимые: {BLOCK_STAGES}")
            # do-while: всегда делаем хотя бы один step(). Это принципиально --
            # если предыдущий блок уже закончился на той же стадии, что и target
            # текущего (например оба 'resid2'), проверка ДО первого шага решила
            # бы, что цель уже достигнута, и блок целиком был бы пропущен.
            while True:
                self.step(state)
                if state.stage == target:
                    break
            return state

        if until is not None and until != "resid2":
            raise ValueError("Частичная стадия блока доступна только при detail=True.")
        if state.stage not in ("embed", "resid2"):
            raise RuntimeError(f"run_block(detail=False) нужно вызывать на границе блока, "
                                f"а не в середине (сейчас stage={state.stage!r}).")

        new_layer = state.layer_idx + 1
        if new_layer >= self.num_layers:
            raise StopIteration(f"Все {self.num_layers} блок(ов) пройдены.")

        block = self.model.transformer.h[new_layer]
        x_in = state.x
        state.layer_idx = new_layer
        self._dispatch("block_in", state, x_in=x_in)
        seq_len = state.ids.shape[1]
        # ВАЖНО: GPT2Block / eager_attention_forward НЕ строит каузальную
        # маску самостоятельно -- если её не передать явно, получится
        # двунаправленное (некаузальное) внимание. В обычном model.forward()
        # эту маску готовит GPT2Model и прокидывает в каждый блок, поэтому
        # при ручном вызове block(...) её нужно собрать и передать так же.
        causal_mask = self._causal_mask(seq_len, x_in.dtype)
        with torch.no_grad():
            out = block(x_in, attention_mask=causal_mask)
        block_out = out[0] if isinstance(out, tuple) else out
        state.x = block_out
        state.cache = {"x_in": x_in, "resid2": block_out}
        state.stage = "resid2"
        self._dispatch("resid2", state)
        self._dispatch("block_out", state, block_out=block_out)
        return state

    # ---------------- быстрый переход к нужному месту ----------------

    def fast_forward(self, state: WalkerState, target_layer: int) -> WalkerState:
        """Быстро (без детализации) прогоняет все блоки СТРОГО ДО target_layer,
        останавливаясь ровно на границе перед ним. Сам target_layer не
        трогает — state остаётся готовым для ручного пошагового step()
        разбора именно этого блока. Это пункт 2 ТЗ: разбираемый этап может
        быть не первым с нуля, но состояние до него считается один раз
        и без лишней детализации.
        """
        if not (0 <= target_layer < self.num_layers):
            raise ValueError(f"В модели только {self.num_layers} блок(ов) (0..{self.num_layers - 1}).")
        if state.layer_idx >= target_layer:
            raise ValueError(
                f"Состояние уже на блоке {state.layer_idx} (или дальше) — "
                f"к блоку {target_layer} «подъехать» заново нельзя. "
                f"Начните новый прогон через walker.start(...).")
        while state.layer_idx < target_layer - 1:
            self.run_block(state, detail=False)
        return state

    def skip_to(self, state: WalkerState, layer_idx: int,
                stage: Optional[str] = None, detail: bool = False) -> WalkerState:
        """Доходит до блока `layer_idx` (предыдущие блоки — быстрым путём,
        без детализации) и, если задана `stage`, детально досчитывает
        нужную под-стадию внутри него. Это и есть "фокус на произвольном
        этапе без пересчёта всего пошагово с нуля" (пункт 2 ТЗ).
        """
        if state.stage == "lm_head":
            raise RuntimeError("State уже завершён (lm_head посчитан).")
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"В модели только {self.num_layers} блок(ов) (0..{self.num_layers - 1}).")
        if state.layer_idx > layer_idx or (state.layer_idx == layer_idx and state.stage == "resid2"):
            raise ValueError(
                f"Состояние уже прошло блок {layer_idx} (сейчас layer_idx="
                f"{state.layer_idx}, stage={state.stage!r}). Назад вернуться нельзя — "
                f"начните заново через walker.start(...).")

        if stage is not None and stage != "resid2":
            detail = True

        while state.layer_idx < layer_idx - 1:
            self.run_block(state, detail=False)

        if state.layer_idx < layer_idx or state.stage != "resid2":
            self.run_block(state, detail=detail, until=stage)

        return state

    # ---------------- финал ----------------

    def finalize(self, state: WalkerState) -> torch.Tensor:
        """ln_f + lm_head. Требует, чтобы ВСЕ блоки уже были пройдены."""
        if state.layer_idx != self.num_layers - 1 or state.stage != "resid2":
            raise RuntimeError(
                f"Чтобы вызвать finalize(), нужно сначала пройти все "
                f"{self.num_layers} блок(ов) (сейчас layer_idx={state.layer_idx}, "
                f"stage={state.stage!r}). Используйте skip_to(state, {self.num_layers - 1}).")
        with torch.no_grad():
            final_norm = self.model.transformer.ln_f(state.x)
            logits = self.model.lm_head(final_norm)
        state.logits = logits
        state.stage = "lm_head"
        self._dispatch("ln_f", state, final_norm=final_norm)
        self._dispatch("lm_head", state, logits=logits)
        return logits

    def decode_topk(self, state: WalkerState, position: int = -1,
                     top_k: int = 15, verbose: bool = True):
        if state.logits is None:
            raise RuntimeError("Сначала вызовите finalize(state).")
        probs = F.softmax(state.logits[0, position], dim=-1)
        k = min(top_k, probs.shape[-1])
        top_probs, top_ids = torch.topk(probs, k)
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_ids.tolist())
        result = list(zip(top_tokens, top_probs.tolist()))
        if verbose:
            print(f"\nПромпт: {state.prompt!r}")
            print(f"Топ-{k} предсказаний следующего токена:\n")
            for tok, p in result:
                display_tok = tok.replace("\u0120", "\u25b8") if tok.startswith("\u0120") else tok
                bar = "\u2588" * int(p * 50)
                print(f"  {display_tok!r:15s} {p:.4f}  {bar}")
        self._dispatch("decode", state, topk=result)
        return result

    # ---------------- сверка с эталоном ----------------

    def verify(self, state: WalkerState, what: str = "auto", verbose: bool = True) -> Optional[float]:
        """max abs diff текущего тензора vs эталонные outputs HF-модели
        (посчитанные один раз в start(), если with_reference=True)."""
        if state.reference is None:
            if verbose:
                print("Эталонные outputs не сохранены (start(..., with_reference=False)).")
            return None
        ref = state.reference
        if what == "auto":
            what = state.stage

        diff = None
        label = None
        if what == "embed":
            ref_hs = ref["hidden_states"][0]
            diff = (state.x.float() - ref_hs.float()).abs().max().item()
            label = "hidden_states[0] (эмбеддинги)"
        elif what in ("resid2", "block_out") and state.layer_idx >= 0 and "resid2" in state.cache:
            if state.layer_idx == self.num_layers - 1:
                # ВАЖНО: для ПОСЛЕДНЕГО блока HF кладёт в hidden_states[-1]
                # результат ПОСЛЕ ln_f (так устроен цикл в GPT2Model.forward:
                # финальный append в all_hidden_states происходит уже после
                # hidden_states = self.ln_f(hidden_states)). Поэтому прямая
                # сверка сырого resid2 с hidden_states[-1] здесь некорректна
                # и покажет расхождение, даже если сам блок посчитан верно.
                # Корректность проверяется сквозным сравнением логитов после
                # finalize() — см. verify(state, what="lm_head").
                if verbose:
                    print("  для последнего блока hidden_states[-1] в HF — это уже "
                          "выход ПОСЛЕ ln_f, сырой resid2 с ним сравнивать нельзя. "
                          "Сверьте логиты после finalize(): verify(state, 'lm_head').")
                return None
            ref_hs = ref["hidden_states"][state.layer_idx + 1]
            diff = (state.cache["resid2"].float() - ref_hs.float()).abs().max().item()
            label = f"hidden_states[{state.layer_idx + 1}] (выход блока {state.layer_idx})"
        elif what == "softmax" and "attn_weights" in state.cache:
            ref_attn = ref["attentions"][state.layer_idx]
            diff = (state.cache["attn_weights"].float() - ref_attn[0].float()).abs().max().item()
            label = f"attentions[{state.layer_idx}]"
        elif what == "lm_head" and state.logits is not None:
            diff = (state.logits.float() - ref["logits"].float()).abs().max().item()
            label = "logits"
        else:
            if verbose:
                print(f"Для стадии {what!r} сверка не определена (нет данных в кэше).")
            return None

        if verbose:
            print(f"  max abs diff vs {label} (эталон): {diff:.3e}")
        return diff


# --------------------------------------------------------------------------
# Рекордеры — модульные визуализаторы / сейверы
# --------------------------------------------------------------------------

class Recorder:
    """Базовый класс. Переопределите `fire()`. Включение/выключение —
    через `enabled`; `stages` ограничивает, на каких стадиях срабатывать
    (пустое множество = на всех)."""

    name = "recorder"
    stages: frozenset = frozenset()

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def should_fire(self, stage: str, layer_idx: int) -> bool:
        return self.enabled and (not self.stages or stage in self.stages)

    def fire(self, *, stage: str, layer_idx: int, state: WalkerState,
              walker: GPT2Walker, **extra) -> None:
        raise NotImplementedError


class AttentionHeatmapRecorder(Recorder):
    """Heatmap внимания по головам (как cell 8 исходного ноутбука)."""
    name = "attention_heatmap"
    stages = frozenset({"softmax"})

    def __init__(self, enabled: bool = True, figsize: Tuple[int, int] = (6, 5)):
        super().__init__(enabled)
        self.figsize = figsize

    def fire(self, *, state, **extra):
        if not _HAS_PLOTTING:
            return
        heads = state.cache["attn_weights"][0].detach().cpu().to(torch.float32).numpy()
        labels = state.tokens
        for i, head_matrix in enumerate(heads):
            plt.figure(figsize=self.figsize)
            plt.title(f"Layer {state.layer_idx} / Attention Head {i}")
            sns.heatmap(head_matrix, xticklabels=labels, yticklabels=labels)
            plt.show()


class HiddenHeatmapRecorder(Recorder):
    """Heatmap MLP-активаций до/после нелинейности и после проекции
    (как cells 21/23/26)."""
    name = "hidden_heatmap"
    stages = frozenset({"mlp_fc", "mlp_act", "mlp_proj"})

    _CACHE_KEY = {"mlp_fc": "mlp_pre_act", "mlp_act": "mlp_post_act", "mlp_proj": "mlp_out"}
    _TITLE = {"mlp_fc": "before mlp activation", "mlp_act": "after mlp activation",
              "mlp_proj": "mlp output (back to embed dim)"}
    _FIGSIZE = {"mlp_fc": (150, 2), "mlp_act": (150, 2), "mlp_proj": (50, 2)}

    def fire(self, *, stage, state, **extra):
        if not _HAS_PLOTTING:
            return
        h = state.cache[self._CACHE_KEY[stage]][0].detach().cpu().to(torch.float32).numpy()
        plt.figure(figsize=self._FIGSIZE[stage])
        plt.title(f"Layer {state.layer_idx}: {self._TITLE[stage]}")
        sns.heatmap(h)
        plt.show()


class NearestTokensRecorder(Recorder):
    """Ближайшие токены словаря по косинусной близости (как cells 16/19)."""
    name = "nearest_tokens"
    stages = frozenset({"resid1", "resid2"})

    def __init__(self, enabled: bool = True, top_k: int = 5, show_gap: bool = True):
        super().__init__(enabled)
        self.top_k = top_k
        self.show_gap = show_gap

    def fire(self, *, stage, state, walker, **extra):
        embed_weight = walker.model.transformer.wte.weight.detach()
        hs = state.cache[stage][0]
        hs_norm = F.normalize(hs, dim=-1)
        emb_norm = F.normalize(embed_weight, dim=-1)
        sims = torch.matmul(hs_norm, emb_norm.T)

        print(f"\n-- ближайшие токены словаря, layer={state.layer_idx}, stage={stage} --")
        for pos in range(hs.shape[0]):
            top_vals, top_idx = torch.topk(sims[pos], min(self.top_k, sims.shape[-1]))
            top_tokens = walker.tokenizer.convert_ids_to_tokens(top_idx.tolist())
            orig = state.tokens[pos]
            line = f"  [{pos}] исходный={orig!r}: "
            line += ", ".join(f"{t!r}={v:.3f}" for t, v in zip(top_tokens, top_vals.tolist()))
            if self.show_gap and len(top_vals) >= 2:
                line += f"  gap={(top_vals[0] - top_vals[1]).item():.4f}"
            print(line)


class ResidualDeltaRecorder(Recorder):
    """Норма входа блока vs норма добавки (attn_out/mlp_out) + косинусная
    близость между ними — насколько сильно attn/mlp двигают residual
    stream (как cells 17/18)."""
    name = "residual_delta"
    stages = frozenset({"resid1", "resid2"})

    _DELTA_KEY = {"resid1": "attn_out", "resid2": "mlp_out"}
    _BASE_KEY = {"resid1": "x_in", "resid2": "resid1"}

    def __init__(self, enabled: bool = True, show_norm_ratio: bool = True, show_cosine: bool = True):
        super().__init__(enabled)
        self.show_norm_ratio = show_norm_ratio
        self.show_cosine = show_cosine

    def fire(self, *, stage, state, **extra):
        base = state.cache[self._BASE_KEY[stage]][0]
        delta = state.cache[self._DELTA_KEY[stage]][0]
        print(f"\n-- residual delta, layer={state.layer_idx}, stage={stage} --")
        for pos in range(base.shape[0]):
            parts = [f"  [{pos}]"]
            if self.show_norm_ratio:
                ratio = delta[pos].norm().item() / base[pos].norm().item()
                parts.append(f"||base||={base[pos].norm().item():.3f} "
                              f"||delta||={delta[pos].norm().item():.3f} ratio={ratio:.3f}")
            if self.show_cosine:
                cos = F.cosine_similarity(base[pos:pos + 1], delta[pos:pos + 1]).item()
                parts.append(f"cos={cos:.4f}")
            print("  ".join(parts))


class ReferenceDiffRecorder(Recorder):
    """Автоматически печатает max abs diff vs эталонные outputs HF-модели
    на контрольных точках (эмбеддинги, softmax-внимание, выход блока, logits)."""
    name = "reference_diff"
    stages = frozenset({"embed", "softmax", "resid2", "lm_head"})

    def fire(self, *, stage, state, walker, **extra):
        walker.verify(state, what=stage, verbose=True)


class TensorSaver(Recorder):
    """Сохраняет указанные тензоры из cache на диск при срабатывании.
    Используется как для одиночных прогонов, так и внутри run_batch()."""
    name = "tensor_saver"

    def __init__(self, enabled: bool = True, stages: Iterable[str] = (),
                 cache_keys: Sequence[str] = (), out_dir: str = "./samples",
                 fmt: str = "pt"):
        super().__init__(enabled)
        self.stages = frozenset(stages)
        self.cache_keys = tuple(cache_keys)
        self.out_dir = Path(out_dir)
        self.fmt = fmt
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def fire(self, *, stage, layer_idx, state, **extra):
        for key in self.cache_keys:
            tensor = state.cache.get(key)
            if tensor is None:
                continue
            fname = f"{self._safe_name(state.prompt)}__L{layer_idx}__{stage}__{key}.{self.fmt}"
            self._save(tensor.detach().cpu(), self.out_dir / fname)

    @staticmethod
    def _safe_name(prompt: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in prompt)[:40]

    def _save(self, tensor: torch.Tensor, path: Path) -> None:
        if self.fmt == "pt":
            torch.save(tensor, path)
        elif self.fmt == "npy":
            import numpy as np
            np.save(str(path.with_suffix(".npy")), tensor.numpy())
        else:
            raise ValueError(f"Неизвестный формат: {self.fmt!r} (используйте 'pt' или 'npy')")


def build_default_walker(model, tokenizer, *, with_heatmaps: bool = True,
                          with_text_diagnostics: bool = True,
                          with_reference_check: bool = True) -> GPT2Walker:
    """Walker с тем же набором рекордеров, что был в исходном ноутбуке —
    каждый включается/выключается через walker.set_enabled(name, bool)."""
    walker = GPT2Walker(model, tokenizer)
    walker.add_recorder(AttentionHeatmapRecorder(enabled=with_heatmaps))
    walker.add_recorder(HiddenHeatmapRecorder(enabled=with_heatmaps))
    walker.add_recorder(NearestTokensRecorder(enabled=with_text_diagnostics))
    walker.add_recorder(ResidualDeltaRecorder(enabled=with_text_diagnostics))
    walker.add_recorder(ReferenceDiffRecorder(enabled=with_reference_check))
    return walker


# --------------------------------------------------------------------------
# Пункт 1: грубый запуск "всех или некоторых" этапов целиком
# --------------------------------------------------------------------------

def run_pipeline(walker: GPT2Walker, prompt: str, *,
                  num_blocks: Optional[int] = None,
                  detail: bool = False,
                  do_lm_head: bool = True,
                  do_decode: bool = True,
                  top_k: int = 15,
                  with_reference: bool = True) -> WalkerState:
    """Запускает этапы модели верхнего уровня: эмбеддинги -> блоки -> lm_head
    -> decode. Какие этапы выполнять, регулируется параметрами:
      num_blocks=0      -> только эмбеддинги
      num_blocks=3       -> первые 3 блока
      num_blocks=None    -> все блоки модели
      do_lm_head=False   -> остановиться после блоков, не считать логиты
      do_decode=False    -> не печатать топ-k предсказаний

    Для подробного пошагового разбора ОДНОГО блока с произвольной точки
    используйте walker.skip_to(...) + walker.step(...) напрямую (пункт 2).
    """
    state = walker.start(prompt, with_reference=with_reference)

    target_blocks = walker.num_layers if num_blocks is None else num_blocks
    target_blocks = min(target_blocks, walker.num_layers)

    while state.layer_idx < target_blocks - 1:
        walker.run_block(state, detail=detail)

    if do_lm_head:
        if state.layer_idx == walker.num_layers - 1:
            walker.finalize(state)
        else:
            print(f"lm_head пропущен: пройдено блоков 0..{state.layer_idx}, "
                  f"а в модели их {walker.num_layers} — для logits нужны все.")

    if do_decode and state.logits is not None:
        walker.decode_topk(state, top_k=top_k)

    return state


# --------------------------------------------------------------------------
# Пункт 4: батч-обработка
# --------------------------------------------------------------------------

def run_batch(walker: GPT2Walker, prompts: Sequence[str], *,
              capture_layer: int, capture_stage: str = "resid2",
              capture_keys: Sequence[str] = ("resid2",),
              save_path: Optional[str] = None,
              detail: bool = False,
              with_reference: bool = False,
              verbose: bool = True) -> Dict[str, Dict[str, torch.Tensor]]:
    """Прогоняет walker на каждом промпте из `prompts`, останавливается на
    (capture_layer, capture_stage) и забирает из state.cache тензоры по
    ключам `capture_keys`. Возвращает dict prompt -> {key: tensor};
    при заданном save_path всё дополнительно сохраняется одним файлом
    через torch.save.

    Пример — захватить attention 3-го слоя для 10 промптов:
        run_batch(walker, prompts, capture_layer=3, capture_stage="softmax",
                  capture_keys=["attn_weights"],
                  save_path="./samples/L3_attention.pt")
    """
    results: Dict[str, Dict[str, torch.Tensor]] = {}
    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"[{i + 1}/{len(prompts)}] {prompt!r}")
        state = walker.start(prompt, with_reference=with_reference)
        walker.skip_to(state, capture_layer, stage=capture_stage, detail=detail)

        captured: Dict[str, torch.Tensor] = {}
        for key in capture_keys:
            if key in state.cache:
                captured[key] = state.cache[key].detach().cpu().clone()
            elif key == "x":
                captured[key] = state.x.detach().cpu().clone()
            elif verbose:
                print(f"  (!) ключ {key!r} не найден в cache на стадии {capture_stage!r}")
        results[prompt] = captured

    if save_path:
        torch.save(results, save_path)
        if verbose:
            print(f"\nСохранено в {save_path}")

    return results
