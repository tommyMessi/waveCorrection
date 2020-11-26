# waveCorrection
OCR Document image deformation correction.复现阿里OCR皱巴巴文档图像形变矫正
此项目启发于阿里云栖的文章《OCR如何读取皱巴巴的文件？深度学习在文档图像形变矫正的应用详解》
出于对此类不开源文章的质疑，证明此方法是否真的有效。
有复杂的图像变换还原可以考虑基于此方法上进行优化。

## 环境(Requirements)
```tensorflow-gpu==1.8```
```Keras==2.0```

## 数据生成
- 波浪变换代码

```python dataGen.py```

## test

```python predict.py```

## train

```python dilatedUnet.py```

## 可视化实例
### 例子🌰1
![1](https://github.com/tommyMessi/waveCorrection/blob/main/data/result_test/result1.png)
![2](https://github.com/tommyMessi/waveCorrection/blob/main/data/result_test/result2.png)
![3](https://github.com/tommyMessi/waveCorrection/blob/main/data/result_test/result3.png)


更多OCR与文档解析，版面分析相关技术 都在微信公众号：hulugeAI   还可以加入专业的OCR文档解析的交流群 业内大佬们等你来
