# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.3-Flash native Megatron conversion and providers."""

from megatron.bridge.models.glm5next.glm53_bridge import GLM53FlashBridge
from megatron.bridge.models.glm5next.glm53_provider import GLM53FlashModelProvider, GLM53FlashTextModelProvider
from megatron.bridge.models.glm5next.modeling_glm53.model import GLM53FlashModel


__all__ = ["GLM53FlashBridge", "GLM53FlashModel", "GLM53FlashModelProvider", "GLM53FlashTextModelProvider"]
