# CPM-TF2Transformer

CPM Fot Transformer

参考了 [qhduan/CPM-LM-TF2](https://github.com/qhduan/CPM-LM-TF2) 的转换代码, 转化成 huggingface 社区 `transformers` 的 `TFGPT2LMHeadModel`

CPM 原REPO：https://github.com/TsinghuaAI/CPM-Generate

原项目首页：https://cpm.baai.ac.cn/

或者可以直接在[colab](https://colab.research.google.com/github/mymusise/CPM-TF2Transformer/blob/main/demo-fp16.ipynb)上运行试试效果 <a href="https://colab.research.google.com/github/mymusise/CPM-TF2Transformer/blob/main/demo-fp16.ipynb">
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>

## 例子

### 1. 依赖

``` 

pip install transformers
pip install jieba #因为原作者在做sentenceprice对文本进行了jieba分词
```

### 2. 初始化模型

模型有 `FP32` 和 `FP16` 两个版本, FP16占用内存小但是在CPU上会比较慢

* (FP32)

``` python
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel.from_pretrained("mymusise/CPM-GPT2")
```

* (FP16)

``` python
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel.from_pretrained("mymusise/CPM-GPT2-FP16")
```

### 3. 文本预处理(仿照原作的处理)

原repo对encode和decode方法做了些特殊处理, 下面仿照原来的方法对 `transformers.XLNetTokenizer` 修改.

``` python
from transformers import XLNetTokenizer

class XLNetTokenizer(XLNetTokenizer):
    translator = str.maketrans(" \n", "\u2582\u2583")

    def _tokenize(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text
```

* FP32

``` python
tokenizer = XLNetTokenizer.from_pretrained('mymusise/CPM-GPT2')
```

* FP16

``` python
tokenizer = XLNetTokenizer.from_pretrained('mymusise/CPM-GPT2-FP16')
```

### 4. 文本生成

``` python
from transformers import TextGenerationPipeline
import jieba

text_generater = TextGenerationPipeline(model, tokenizer)

texts = [
    '今天天气不错',
    '天下武功, 唯快不',
    """
    我们在火星上发现了大量的神奇物种。有神奇的海星兽，身上是粉色的，有5条腿；有胆小的猫猫兽，橘色，有4条腿；有令人恐惧的蜈蚣兽，全身漆黑，36条腿；有纯洁的天使兽，全身洁白无瑕，有3条腿；有贪吃的汪汪兽，银色的毛发，有5条腿；有蛋蛋兽，紫色，8条腿。

    请根据上文，列出一个表格，包含物种名、颜色、腿数量。
    |物种名|颜色|腿数量|
    |亚古兽|金黄|2|
    |海星兽|粉色|5|
    |猫猫兽|橘色|4|
    |蜈蚣兽|漆黑|36|
    """
]

for text in texts:
    token_len = len(tokenizer._tokenize(text))
    print(text_generater(text, max_length=token_len + 15, top_k=1, use_cache=True, prefix='')[0]['generated_text'])
    print(text_generater(text, max_length=token_len + 15, do_sample=True, top_k=5)[0]['generated_text'])
```

### 5. 效果:

![avatar](example-cpm.png)

## Changelog

* 2021-02-24: 修复模型转换问题, 更新decode和encode方法
