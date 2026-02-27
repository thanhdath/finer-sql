![visitors](https://visitor-badge.laobi.icu/badge?page_id=thanhdath.finer-sql)

# FINER-SQL: Boosting Small Language Models for Text-to-SQL with Fine-Grained Execution Feedback and Cost-Efficient Rewards

**Models:** 👉 [https://huggingface.co/collections/griffith-bigdata/finer-sql](https://huggingface.co/collections/griffith-bigdata/finer-sql)

### Citation
```
@inproceedings{finersql,
  author       = {Thanh Dat Hoang and Thanh Trung Huynh and Matthias Weidlich and Thanh Tam Nguyen and Tong Chen and Hongzhi Yin and Quoc Viet Hung Nguyen},
  title        = {Boosting Small Language Models for Text-to-SQL with Fine-Grained Execution Feedback and Cost-Efficient Rewards},
  booktitle    = {ICDE},
  publisher    = {IEEE},
  year         = {2026},
}
```

---

FINER-SQL introduces **dense, interpretable rewards** to train **small language models (≤3B)** for Text-to-SQL via **Group Relative Policy Optimization (GRPO)**.  
Beyond from Format Reward and Execution Reward, it combines:
- **Memory Reward** — semantic alignment with verified reasoning traces  
- **Atomic Reward** — atomic operation-level SQL overlap for structural feedback

This helps solving the sparse reward issue of reinforcement learning in Text-to-SQL.

✅ Achieves 67.5% EX on BIRD when training only on BIRD train, and 85% EX on Spider using only a 3B model.  
⚡ Runs efficiently on a single 12-24 GB GPU.

---

**We are cleaning and updating the code for easy to use.**


-----------
**Backup Statistics**

![Visitors](https://margherita-gustatory-zane.ngrok-free.dev/badge/thanhdath%2Ffiner-sql.svg?ngrok-skip-browser-warning=true)

