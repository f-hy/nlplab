# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
config settings, will be used in finetune.py
"""

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .bert_model import BertConfig


bert_optimizer_cfg = edict({
    'AdamWeightDecayDynamicLR': edict({
        'learning_rate': 2e-5,
        'end_learning_rate': 1e-7,    
        'power': 1.0,
        'weight_decay': 1e-5,
        'eps': 1e-6,
    }),
    'Lamb': edict({
        'start_learning_rate': 2e-5,
        'end_learning_rate': 1e-7,     
        'power': 1.0,
        'weight_decay': 0.01,
        'decay_filter': lambda x: False,
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }), 
})

tag_to_index = {
    "O": 0,
    "S_address": 1,
    "B_address": 2,
    "M_address": 3,
    "E_address": 4,
    "S_book": 5,
    "B_book": 6,
    "M_book": 7,
    "E_book": 8,
    "S_company": 9,
    "B_company": 10,
    "M_company": 11,
    "E_company": 12,
    "S_game": 13,
    "B_game": 14,
    "M_game": 15,
    "E_game": 16,
    "S_government": 17,
    "B_government": 18,
    "M_government": 19,
    "E_government": 20,
    "S_movie": 21,
    "B_movie": 22,
    "M_movie": 23,
    "E_movie": 24,
    "S_name": 25,
    "B_name": 26,
    "M_name": 27,
    "E_name": 28,
    "S_organization": 29,
    "B_organization": 30,
    "M_organization": 31,
    "E_organization": 32,
    "S_position": 33,
    "B_position": 34,
    "M_position": 35,
    "E_position": 36,
    "S_scene": 37,
    "B_scene": 38,
    "M_scene": 39,
    "E_scene": 40,
    "<START>": 41,
    "<STOP>": 42
}
