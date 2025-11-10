# FINER-SQL: Boosting Small Language Models for Text-to-SQL with Fine-Grained Execution Feedback and Cost-Efficient Rewards

**Models:** 👉 [https://huggingface.co/collections/griffith-bigdata/finer-sql](https://huggingface.co/collections/griffith-bigdata/finer-sql)

---

FINER-SQL introduces **dense, interpretable rewards** to train **small language models (≤3B)** for Text-to-SQL via **Group Relative Policy Optimization (GRPO)**.  
It combines:

- 🧠 **Memory Reward** — semantic alignment with verified reasoning traces  
- ⚙️ **Atomic Reward** — atomic operation-level SQL overlap for structural feedback  

✅ Achieves 67.5% EX on BIRD when training only on BIRD train, and 85% EX on Spider using only a 3B model.  
⚡ Runs efficiently on a single 12-24 GB GPU.

---

**We are cleaning and updating the code for easy to use.**
