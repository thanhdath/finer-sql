# FINER-SQL: Boosting Small Language Models for Text-to-SQL with Fine-Grained Execution Feedback

**Models:** 👉 [https://huggingface.co/collections/griffith-bigdata/finer-sql](https://huggingface.co/collections/griffith-bigdata/finer-sql)

---

FINER-SQL introduces **dense, interpretable rewards** to train **small language models (≤3B)** for Text-to-SQL via **Group Relative Policy Optimization (GRPO)**.  
It combines:

- 🧠 **Memory Reward** — semantic alignment with verified reasoning traces  
- ⚙️ **Atomic Reward** — atomic operation-level SQL overlap for structural feedback  

✅ Achieves 67.5% EX on BIRD and 85% EX on Spider using only a 3B model.  
⚡ Runs efficiently on a single 24 GB GPU.

---
