# GLM-5.x

Ветка под сборку образа GLM-5.x (verl + sglang + Megatron-LM + slime + Megatron-Bridge).

Раньше Bridge подключался клоном апстрима NVIDIA-NeMo с пином
`d0c6228a2a832f566dd44a3a179b3136613c11b7` и cherry-pick
`44c871b4eab107028933ea1a3aaa42dacc260c1c` («fix: dequantize GLM 5 FP8 weights on import»,
PR #5851). Этот коммит влит в апстрим 02.09.2026 и уже входит в историю ветки, поэтому
cherry-pick и сверка sha256 патча больше не нужны — образ ставит Bridge отсюда.

Проверено на срезе:
- `src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py` содержит
  `maybe_modify_loaded_hf_weight` и `_maybe_dequantize_fp8` (ключ `<param>_scale_inv`);
- между старым пином и веткой `glm_moe_dsa` трогали ровно два коммита: сам фикс FP8
  и `feat(model): add Muse Glimmer support` (к GLM-пути отношения не имеет).

Bridge по-прежнему подключается исходниками через PYTHONPATH, а не пакетом: метаданные
опубликованного колеса тянут конфликтующий transformers+NeMo-стек.
