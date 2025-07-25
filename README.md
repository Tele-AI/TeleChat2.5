<div align="center">
<h1>
  TeleChat2.5
</h1>
</div>


<p align="center">
   🦉 <a href="https://github.com/Tele-AI/TeleChat2.5" target="_blank">github</a> • 🤗 <a href="https://huggingface.co/Tele-AI" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/TeleAI" target="_blank">ModelScope</a> • 🐾 <a href="https://gitee.com/Tele-AI/TeleChat2.5" target="_blank">gitee</a> • 💬 <a href="https://github.com/Tele-AI/Telechat/blob/master/images/wechat.jpg" target="_blank">WeChat</a>
</p>

# 目录

- [模型介绍](#模型介绍)
- [效果评测](#效果评测)
- [模型推理](#模型推理)
- [国产化适配](#国产化适配)
- [声明、协议、引用](#声明协议引用)

# 模型介绍

**TeleChat2.5** 是 **TeleChat** 系列新版通用问答模型，由中国电信人工智能研究院（**TeleAI**）基于国产算力研发训练，包括了 **TeleChat2.5-35B** 与 **TeleChat2.5-115B**。TeleChat2.5 基于最新强化的 TeleBase2.5 系列模型进行训练，在理科、通用问答、Function Call等任务上有显著的效果提升。TeleChat2.5 的微调方法延续了 TeleChat2 系列，具体请参考 [TeleChat2](https://github.com/Tele-AI/TeleChat2)。

### 训练策略
#### 数据

- 为了提高模型训练数据的数量和质量，TeleChat2.5 在训练过程中采用了大量理科学科和编程领域的合成数据。在合成过程中，为了减少错误信息的引入，主要以基于知识点或知识片段的教育类知识合成为主。


#### 基础模型训练

- TeleChat2.5 采用了多阶段课程学习策略，在训练过程中逐步提升理科和编程类高密度知识数据的比例。每个训练阶段都使用比前一阶段质量更高、难度更大的数据，以实现持续的模型优化。  

- 在最终训练阶段，为了平衡模型在各个维度的能力表现，我们选取了不同训练阶段效果较优的多个模型，并基于各模型的综合表现进行参数加权融合，其中权重分配与模型性能呈正相关。

#### 后训练阶段
我们采用分阶段优化的模型训练策略：

- 融合优化阶段：整合复杂推理与通用问答能力，针对语言理解、数理逻辑等薄弱任务进行解构重组。通过重构任务框架并融合多维度解题思路，生成优化后的通用答案集。此阶段答案长度会适度增加，并基于优化数据实施微调训练。

- 能力强化阶段：针对数理逻辑与编程类任务，通过注入结构化解题思路，结合基于规则的强化学习奖励机制，显著提升模型对复杂任务的理解与处理能力。

- 泛化提升阶段：面向安全合规、指令响应、函数调用、数学推理、代码生成等十余种任务类型进行系统性强化学习增强，全面提升模型的通用任务处理能力。

### 模型下载
| 模型版本             | 下载链接                                                          | 
|------------------|--------------------------------------------------------------------|
| TeleChat2.5-35B  | [modelscope](https://modelscope.cn/models/TeleAI/TeleChat2.5-35B)  |
| TeleChat2.5-115B | [modelscope](https://modelscope.cn/models/TeleAI/TeleChat2.5-115B) |

# 效果评测
| 模型               | MATH-500 | AlignBench | BFCL(avg v1&v2) |
|------------------|----------|------------|-----------------|
| Qwen2.5-32B      | 82       | 7.39       | 81.11           |
| Qwen2.5-72B      | 82       | 7.62       | 79.15           |
| Qwen3-32B（通用）    | 83       | 8.23       | 81.84           |
| GPT-4o-1120      | 75       | 7.49       | 78.66           |
| TeleChat2-35B    | 65       | 6.97       | 75.32           |
| TeleChat2-115B   | 75       | 7.56       | 77.47           |
| TeleChat2.5-35B  | 85        | 7.73       | 78.28           |
| TeleChat2.5-115B | 87       | 7.93       | 83.39           |


# 模型推理

TeleChat2.5 系列模型支持使用 `transformers` 库进行推理，示例如下：


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("TeleChat2.5/TeleChat2.5-35B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "TeleChat2.5/TeleChat2.5-35B",
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
prompt = "生抽和酱油的区别是什么？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

#### ModelScope
TeleChat2.5 系列模型支持使用 ModelScope 推理，示例如下：
```python
import os
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
tokenizer = AutoTokenizer.from_pretrained('TeleChat2.5/TeleChat2.5-35B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('TeleChat2.5/TeleChat2.5-35B', trust_remote_code=True, device_map="auto",
                                                  torch_dtype=torch.bfloat16)
prompt = "生抽与老抽的区别？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```


### vLLM 推理

TeleChat2.5 支持使用 [vLLM](https://github.com/vllm-project/vllm) 进行部署与推理加速，示例如下:
##### 离线推理
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("TeleChat2.5/TeleChat2.5-35B", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.01, max_tokens=8192)
llm = LLM(model="TeleChat2.5/TeleChat2.5-35B", trust_remote_code=True, tensor_parallel_size=4, dtype="bfloat16")

prompt = "生抽和酱油的区别是什么？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = llm.generate([text], sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

##### OpenAI 兼容的 API 服务
您可以借助 vLLM，构建一个与 OpenAI API 兼容的 API 服务。请按照以下所示运行命令：
```
vllm serve TeleChat2.5/TeleChat2.5-35B \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-custom-all-reduce
```
然后，您可以与 TeleChat2.5 进行对话：
```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
chat_response = client.chat.completions.create(
    model="TeleChat2.5/TeleChat2.5-35B",
    messages=[
        {"role": "user", "content": "生抽和酱油的区别是什么？"},
    ],
    temperature=0.0,
    max_tokens=8192,
    extra_body={
        "repetition_penalty": 1.01,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    },
)
print("Chat response:", chat_response)
```

# 国产化适配

TeleChat2.5系列模型均进行了**国产化算力适配**，具体信息可见
1. <a href="https://modelers.cn/models/MindSpore-Lab/TeleChat2.5-35B" target="_blank">MindSpore-Lab/TeleChat2.5-35B</a>
2. <a href="https://modelers.cn/models/MindSpore-Lab/TeleChat2.5-115B" target="_blank">MindSpore-Lab/TeleChat2.5-115B</a>

# 声明、引用

### 声明

我们在此声明，不要使用 TeleChat2.5 系列模型及其衍生模型进行任何危害国家社会安全或违法的活动。同时，我们也要求使用者不要将 TeleChat2.5 系列模型用于没有安全审查和备案的互联网服务。我们希望所有使用者遵守上述原则，确保科技发展在合法合规的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用 TeleChat2.5 系列开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

### 引用

如需引用我们的工作，请使用如下 reference:

```
@misc{wang2025technicalreporttelechat2telechat25,
      title={Technical Report of TeleChat2, TeleChat2.5 and T1}, 
      author={Zihan Wang and Xinzhang Liu and Yitong Yao and Chao Wang and Yu Zhao and Zhihao Yang and Wenmin Deng and Kaipeng Jia and Jiaxin Peng and Yuyao Huang and Sishi Xiong and Zhuo Jiang and Kaidong Yu and Xiaohui Hu and Fubei Yao and Ruiyu Fang and Zhuoru Jiang and Ruiting Song and Qiyi Xie and Rui Xue and Xuewei He and Yanlei Xue and Zhu Yuan and Zhaoxi Zhang and Zilu Huang and Shiquan Wang and Xin Wang and Hanming Wu and Mingyuan Wang and Xufeng Zhan and Yuhan Sun and Zhaohu Xing and Yuhao Jiang and Bingkai Yang and Shuangyong Song and Yongxiang Li and Zhongjiang He and Xuelong Li},
      year={2025},
      eprint={2507.18013},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.18013}, 
}
```

