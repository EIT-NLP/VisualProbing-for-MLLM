# Multimodal Language Models See Better When They Look Shallower

[![arXiv](https://img.shields.io/badge/arXiv-2504.21447-b31b1b.svg)](https://arxiv.org/abs/2504.21447)

## 📖 Introduction

Multimodal Large Language Models (MLLMs) usually rely on the last or penultimate layer of CLIP-ViT for visual inputs, forming a default **deep-layer-first paradigm**.  
But are deep features always optimal? Have the representational powers of shallow and middle layers been underestimated?

<div align="center">
  <img src="assets/images/figure1.png" alt="Visual Layer Analysis" width="80%">
  <p><em>Figure 1: Layer-wise visual feature analysis in CLIP-ViT</em></p>
</div>

In this work, we analyze **layer-wise representational similarity** in CLIP-ViT and its correlation with downstream performance.  
- We clearly divide shallow, middle, and deep feature spaces.  
- Our study spans **models from 1.4B to 7B**, **datasets from 665K to 1M**, covering **10 benchmarks and 60+ tasks**.  

🔑 **Key Findings:**  
- **Deep features** → stronger in semantically intensive tasks (e.g., OCR).  
- **Shallow & middle features** → better for fine-grained perception tasks (counting, localization, object recognition).  
- A simple **cross-layer fusion method** effectively harnesses shallow and middle features, further validating their importance.


This work supports our main claim:  
**“Multimodal Language Models See Better When They Look Shallower.”**  
It also offers both theoretical foundations and practical guidelines for visual layer selection and fusion in future MLLMs.

---



## ✅ TODO List
- [ ] Release training & evaluation code
- [ ] Provide documentation & usage examples 


🚀 _Code will be released soon at_ 👉 [https://github.com/EIT-NLP/VisualProbing-for-MLLM](https://github.com/EIT-NLP/VisualProbing-for-MLLM)

---

## 📚 Citation
```bibtex
@misc{chen2025rethinkingvisuallayerselection,
      title={Rethinking Visual Layer Selection in Multimodal LLMs}, 
      author={Haoran Chen and Junyan Lin and Xinhao Chen and Yue Fan and Xin Jin and Hui Su and Jianfeng Dong and Jinlan Fu and Xiaoyu Shen},
      year={2025},
      eprint={2504.21447},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.21447}, 
}
```
