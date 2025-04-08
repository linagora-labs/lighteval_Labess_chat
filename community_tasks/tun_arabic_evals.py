# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import random
import re
from typing import Any, Dict, List, Optional, Union


from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mmlu_tun_arabic(line, task_name: str = None):    
    topic = line["subject"]
    question = line["question"]
    instruction = f"هاذا سؤال متعدد الخيارات (مع الجواب متاعو) على {topic.replace('_', ' ')}. \n\n"
    choices = line["choices"]
    # Build the query and determine the gold_index in a single pass
    query = f"{instruction}السؤال: {question}\n"    
    query += "\n اختر الجواب من بين هذه الخيارات فقط:"
        
    gold_index = int(line["answer"])  
    # Show all choises
    for i, choice in enumerate(choices):
        query += f"{i}. {choice}\n"

    query += "\n\nالإجابة:"         
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,       
    )

ARABIC_TUN_MMLU_SUBSETS = [
    "accounting","arabic_language","arabic_language_(general)","arabic_language_(grammar)","biology","civics","computer_science","driving_test","economics","general_knowledge",
    "geography","global_facts","high_school_european_history","high_school_geography","high_school_government_and_politics","high_school_psychology",
    "high_school_statistics","high_school_world_history","history","human_aging","international_law","islamic_studies","jurisprudence","law","logical_fallacies","management",
    "management_ar","marketing", "math","moral_disputes","moral_scenarios", "natural_science","nutrition","philosophy","philosophy_ar","physics","political_science","professional_law",
    "professional_psychology","public_relations","security_studies","social_science","sociology", "world_religions",    
]

class CustomTUNArabicMMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=mmlu_tun_arabic,
            hf_repo="linagora/TunisianMMLU",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["custom"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )

TASKS_TABLE = [
    CustomTUNArabicMMLUTask(name=f"TunisianMMLU:{subset}", hf_subset=subset) for subset in ARABIC_TUN_MMLU_SUBSETS
]

if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
   
