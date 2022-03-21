## 引言

QoS预测是服务计算中非常热门的话题，随着研究的深入，越来越多高效、准确的QoS预测方法被提出。但众多方法实现标准各异，这导致了当一个新方法提出时，很难在同一尺度下与先前的方法进行公平地竞争。本项目旨在复现历来被人熟知的QoS预测方法，并统一初始化参数、统一训练数据结构、统一训练方法，构建一个内容丰富、使用简单的QoS预测算法库。

## 代办事项


| Memory-Based | 完成情况 | 论文 | 公式 |  |
| -------------- | ---------- | -------------- | -------------- | -------------- |
| UMEAN        | ✅       |  |  |  |
| IMEAN        | ✅       |  |  |  |
| UPCC         | ✅       | [Shao L, Zhang J, Wei Y, et al. Personalized qos prediction forweb services via collaborative filtering[C]//Ieee international conference on web services (icws 2007). IEEE, 2007: 439-446.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4279629) |  |  |
| IPCC         | ✅ | [MLASarwar B, Karypis G, Konstan J, et al. Item-based collaborative filtering recommendation algorithms[C]//Proceedings of the 10th international conference on World Wide Web. 2001: 285-295.](https://dl.acm.org/doi/pdf/10.1145/371920.372071) |  |  |
| WSRec(UIPCC) |          |  |  |  |
|  | |  |  |  |
| NRCF         |          |  |  |  |
| RACF         |          |  |  |  |
|              |          |  |  |  |


| Model-Based | 完成情况 | 论文 | 公式 |
| ------------- | ---------- | :------------ | ------------- |
| MF    | ✅       | Koren Y, Bell R, Volinsky C. Matrix factorization techniques for recommender systems[J]. Computer, 2009, 42(8): 30-37. | $ \min _{\mathbf{p}, \mathbf{q}} \frac{1}{2} \sum_{(u, i) \in \mathbf{O}}\left\|r_{u, i}-\mathbf{p}_{u} \mathbf{q}_{i}^{T}\right\|^{2}+\frac{1}{2} \lambda\left(\left\|\mathbf{p}_{u}\right\|^{2}+\left\|\mathbf{q}_{i}\right\|^{2}\right)$ |
| PMF | ✅ |  | $E=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{M} I_{i j}\left(R_{i j}-U_{i}^{T} V_{j}\right)^{2}+\frac{\lambda_{U}}{2} \sum_{i=1}^{N}\left\|U_{i}\right\|_{F r o}^{2}+\frac{\lambda_{V}}{2} \sum_{j=1}^{M}\left\|V_{j}\right\|_{F r o}^{2}$ |
| NMF | ✅ | Lee D D, Seung H S. Learning the parts of objects by non-negative matrix factorization[J]. Nature, 1999, 401(6755): 788-791. | $ \begin{aligned} &\min _{\mathbf{p}, \mathbf{q}} \frac{1}{2} \sum_{(u, i) \in \mathbf{O}}\left\|r_{u, i}-\mathbf{p}_{u} \mathbf{q}_{i}^{T}\right\|^{2} \\ &\text { s.t. } \mathbf{p}_{u, \cdot}>0, \mathbf{q}_{i, \cdot}>0 \end{aligned}$ |
| MLP       |       |  |  |
| NewMF | | | |
| GMF         |          |  |  |
|             |          |  |  |


| Federated-Based | 完成情况 | 算法介绍/论文/公式 |
| ----------------- | ---------- | -------------------- |
| FedMF           | ✅       |                    |
| FedNMF |          |                    |
|                 |          |                    |


| 杂项         | 完成情况 |  |
| -------------- | ---------- | -- |
| 注释         |          |  |
| 训练日志     |          |  |
| 复杂度优化   | :stopwatch: |  |
| 训练数据保存 | ✅       |  |
| 支持GPU      | ✅       |  |
| 训练可视化 | :stopwatch: |  |
|              |          |  |

## Baseline

## Reference

[pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)



