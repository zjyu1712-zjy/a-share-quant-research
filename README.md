# A-share Quant Research

A Python research framework for point-in-time A-share universes, factor screening, realistic portfolio simulation, attribution, and paper-trading records.

一个面向个人研究者的 A 股量化研究框架：先核实数据，再筛选因子，再验证组合，保留失败结果与模拟盘记录。

**当前定位：研究代码，不是已验证盈利的自动交易系统。** 本仓库不连接券商下单；当前 v2 的三个候选组合均未通过既定研究门槛。年化 10% 和最大回撤 10% 是研究目标，不是收益或损失上限承诺。

## 核心内容

- 历史时点上证综指可交易 A 股研究池，默认排除科创板；保留快照与剔除原因。
- 低波动、行业中性 1/PB、剔除最近一个月的中期动量等因子筛选。
- 月频 Top 10、排名缓冲、行业限制和组合回撤控制的独立研究。
- 信号之后的交易日开盘执行，包含佣金、最低佣金、税费、滑点、整手和成交限制。
- 走步验证、收益归因、组合失败归因，以及独立的 v1 模拟盘记录模块。
- 无需 API 密钥或真实行情文件即可运行的合成数据回归测试。

v2 固定策略、口径及已知限制见 [研究说明](docs/RESEARCH.md)。科创板开关默认关闭；打开开关不等于具备券商权限，也不代表允许实盘部署。

## 快速开始：先运行离线测试

建议使用 Python 3.12。以下命令不下载行情、不调用新闻 API，也不生成真实买入建议。

```bash
git clone https://github.com/zjyu1712-zjy/a-share-quant-research.git
cd a-share-quant-research
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

macOS 也可以运行 `./setup_mac.command` 安装依赖，再运行上述测试。该安装脚本会创建空白 `.env` 模板，不要求提供新闻密钥。Windows 可使用 `py -3.12 -m venv .venv` 与 `.venv\Scripts\Activate.ps1`；macOS `.command` 启动器不适用于 Windows。持续集成验证 Ubuntu/macOS；Windows 尚未纳入本发布版自动验证。

## v2 研究流程

真实研究需要用户自行获取数据，数据源的可用性及使用条款由其提供方决定。历史股票池并集下载可能耗时较长；下载器支持断点续跑，不应为了加速而只选已有缓存覆盖的股票。

在项目根目录、已激活虚拟环境的终端中，依次运行：

```bash
# 1. 构建历史股票池并审计
python -m src.sse_composite_universe --start 2020-01-01 --end 2025-05-12
python -m src.v2_monthly_validation --start 2020-01-01 --end 2025-05-12

# 2. 准备含因子预热期的价格和估值数据
python -m src.v2_factor_data --start 2019-01-01 --end 2025-05-12

# 3. 仅在数据审计通过后，筛选独立因子
python -m src.v2_factor_screening --start 2020-01-01 --end 2025-05-12

# 4. 验证组合，并检查组合为何失败
python -m src.v2_factor_portfolio_validation --start 2020-01-01 --end 2025-05-12
python -m src.v2_lowvol_failure_attribution --start 2020-01-01 --end 2025-05-12
```

不要跳过失败的数据或因子准入检查。输出写入本地 `data/`，不会被 Git 跟踪。原始数据库、价格/估值缓存、新闻全文、个人模拟盘账本与本机环境均不随仓库分发；公开代码不意味着拥有数据再分发权。

macOS 的等价快捷命令可通过 `./run_mac.sh help` 查看。**1–4 号快捷入口属于旧版 v1，不是上述 v2 月频策略入口。** v2 研究命令不会升级或覆盖 v1 模拟盘。

## 当前研究结果

下表来自已完成的历史组合复验，连续评价区间为 2022-01-01 至 2025-05-12（信号与成交仍按交易日规则执行）：

| 候选组合 | 年化收益 | 最大回撤 | 准入结论 |
| --- | ---: | ---: | --- |
| 60 日低波动 | -1.11% | -10.40% | 未通过 |
| 行业中性 1/PB | -1.59% | -15.15% | 未通过 |
| 低波动 50% + 价值 50% | -1.35% | -15.35% | 未通过 |

可核对的汇总表见 [组合结果](docs/research/portfolio_continuous_results.csv) 和 [逐步归因](docs/research/lowvol_ablation_results.csv)。这些是历史研究记录，不是重新运行本次发布所得到的新收益，也不是真正未来未见的业绩。因子筛选和组合复验使用重叠历史区间，走步窗口不能消除该层研究选择偏差。

## 目录

```text
src/             数据、因子、执行、验证、归因与模拟盘模块
tests/           合成数据和临时目录驱动的回归测试
config/          默认关闭的科创板配置
MyQuantAgent/    旧版兼容用静态映射；不是 v2 股票池
docs/            方法、边界与少量聚合结果
```

新闻模块为可选实验功能：会把抓取的新闻发送给配置的模型服务商，并可能产生 API 费用。使用前填写 `.env` 中的密钥及有效模型名，确认内容使用和传输权限；纯价格研究不需要该服务。

傅立叶去噪和 Parquet 存储优化尚未包含在本发布版中。未来若加入，必须分别验证因果性、边界效应、信息损失和样本外增量，不能把平滑后的曲线等同于更有效的策略。

## 来源与许可

项目由既有选股研究代码演进，保留了 `MyQuantAgent` 来源说明。当前**未添加新的开源许可证**，也不宣称独占原始代码权利。公开可见不等于授予任意复制、修改或商业再分发许可；详见 [来源与许可状态](LICENSE_STATUS.md)。
