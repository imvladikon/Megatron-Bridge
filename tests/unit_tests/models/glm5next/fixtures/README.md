# Published metadata fixtures

Retrieved 2026-09-12. Config files are complete source metadata; no weights are
included or loaded by the configuration tests.

- `flash_full.json`: `zai-org/GLM-5.3-Flash`, revision
  `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, `config.json`.
- `flash_tiny.json`: `inference-optimization/GLM-5.3-Flash-0.1B-A0.1B`, revision
  `8311399447eba9c9b215e3209ab6f25e59c7d21e`, `config.json`.
- `flash_tiny_mla_shapes.json`: DSA layer 3 shapes from that tiny revision's
  `model.safetensors` header. Retrieved using HTTP byte ranges for the 8-byte
  header length and 27,456-byte JSON header only; tensor data were not read.

The tiny config retains `qk_head_dim=256`, but Transformers v5.16.1 builds MLA
using `qk_nope_head_dim + qk_rope_head_dim = 64 + 0`. The stored projection
shapes independently confirm this. Do not "repair" the fixture: it is the
published case the adapter must interpret faithfully.
