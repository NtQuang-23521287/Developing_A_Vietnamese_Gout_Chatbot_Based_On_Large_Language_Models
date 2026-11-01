Split: dev, n = 114

| Chỉ số | Qwen-Base | Qwen-SFTv1 |
| --- | --- | --- |
| Reward (TB trọng số) | 0.6631 | 0.3588 |
| Fatal | 0 | 1 |
| Reward CI95 | [0.605, 0.723] | [0.270, 0.445] |
| faithfulness (0–1) | 0.812 | 0.730 |
| completeness (0–1) | 0.765 | 0.412 |
| relevance (0–1) | 0.985 | 0.903 |
| context_recall (0–1) | 0.965 | 0.943 |
| patient_utility (0–1) | 0.857 | 0.626 |
| refusal_appropriateness (0–1) | 1.000 | 0.897 |
| Rate hallucination_level>=2 | 0.164 | 0.081 |
| Rate omission_risk>=2 | 0.152 | 0.634 |
| Rate safety_risk>=2 | 0.068 | 0.210 |
| Rate safety_risk>=3 | 0.000 | 0.015 |

| So sánh | Chênh reward (A−B) | CI95 | Wilcoxon p | Kết luận |
| --- | --- | --- | --- | --- |
| Qwen-Base vs Qwen-SFTv1 | +0.3042 | [+0.2079, +0.4050] | 0.0000 | Qwen-Base tốt hơn |
