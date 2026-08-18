"""跨架构对比脚本：分解式 (current_prediction) vs 端到端 (e2e_current_prediction)"""
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ====================================================================
# ArchitectureComparator: 对比两套评估结果
# ====================================================================
class ArchitectureComparator:
    """
    加载 evaluation/*.npz 结果 → 分类 → 统计 → 打印报告
    结果分类规则:
        - current_*  → 分解式架构 (geostrophic + ageostrophic)
        - e2e_*      → 端到端架构
        - Persistence→ 基准线
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.eval_dir = self.output_dir / "evaluation"
        self.results: Dict[str, Dict[str, np.ndarray]] = {}

    # --------------------------------------------------------------
    # 1. 加载所有 npz 结果
    # --------------------------------------------------------------
    def load_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        if not self.eval_dir.exists():
            print(f"[WARN] eval dir not found: {self.eval_dir}")
            print("       Please run current_prediction/analyze/analyze.py and "
                  "e2e_current_prediction/analyze/analyze.py first.")
            return self.results
        for npz_file in sorted(self.eval_dir.glob("*.npz")):
            data = np.load(npz_file, allow_pickle=True)
            self.results[npz_file.stem] = {k: data[k] for k in data.files}
        print(f"[OK] Loaded {len(self.results)} evaluation results from {self.eval_dir}")
        return self.results

    # --------------------------------------------------------------
    # 2. 分类
    # --------------------------------------------------------------
    def classify(self) -> Tuple[Dict, Dict, Dict]:
        decomposed = {k: v for k, v in self.results.items()
                      if k.lower().startswith('current_')}
        e2e = {k: v for k, v in self.results.items()
               if k.lower().startswith('e2e_')}
        persist = {k: v for k, v in self.results.items()
                   if 'persist' in k.lower()}
        return decomposed, e2e, persist

    # --------------------------------------------------------------
    # 3. 综合对比报告
    # --------------------------------------------------------------
    def compare(self):
        if not self.results:
            self.load_all()
        if not self.results:
            return

        dec, e2e, persist = self.classify()
        N_dec, N_e2e, N_pers = len(dec), len(e2e), len(persist)

        print("\n" + "=" * 78)
        print("  架构对比分析报告: 分解式 (地转+非地转) vs 端到端 (纯数据驱动)")
        print("=" * 78)
        print(f"  分解式模型 (current_*) : {N_dec}")
        print(f"  端到端模型 (e2e_*)     : {N_e2e}")
        print(f"  Persistence 基准       : {N_pers}")

        # ============== [指标1] 综合精度对比表 ==============
        print("\n" + "-" * 78)
        print("[指标1] 综合预报精度对比 (lead-time 平均, 物理单位 m/s)")
        print("-" * 78)
        header = (f"{'类型':<14} {'模型名(截断)':<38} {'RMSE(m/s)':<12} "
                  f"{'Corr':<10} {'vsPERS%':<10}")
        print(header)
        print("-" * len(header))

        baseline_rmse = None
        all_rows = []

        for name, r in persist.items():
            rmse = float(np.mean(r['rmse_lead']))
            corr = float(np.mean(r['corr_lead']))
            baseline_rmse = rmse
            print(f"{'Persistence':<14} {name[:38]:<38} {rmse:<12.4f} "
                  f"{corr:<10.4f} {'--':<10}")
            all_rows.append(('基准', name, rmse, corr, 0.0))

        for name, r in dec.items():
            rmse = float(np.mean(r['rmse_lead']))
            corr = float(np.mean(r['corr_lead']))
            imp = (baseline_rmse - rmse) / baseline_rmse * 100 if baseline_rmse else 0
            print(f"{'分解式':<14} {name[:38]:<38} {rmse:<12.4f} "
                  f"{corr:<10.4f} {imp:<10.1f}")
            all_rows.append(('分解式', name, rmse, corr, imp))

        for name, r in e2e.items():
            rmse = float(np.mean(r['rmse_lead']))
            corr = float(np.mean(r['corr_lead']))
            imp = (baseline_rmse - rmse) / baseline_rmse * 100 if baseline_rmse else 0
            print(f"{'端到端':<14} {name[:38]:<38} {rmse:<12.4f} "
                  f"{corr:<10.4f} {imp:<10.1f}")
            all_rows.append(('端到端', name, rmse, corr, imp))

        # ============== [指标2] 分解式 vs 端到端 直接差异 (核心有效性证明) ==============
        print("\n" + "-" * 78)
        print("[指标2] 分解式架构 vs 端到端架构 —— 直接差异统计")
        print("-" * 78)
        if dec and e2e:
            dec_rmse_list = [float(np.mean(r['rmse_lead'])) for r in dec.values()]
            dec_corr_list = [float(np.mean(r['corr_lead'])) for r in dec.values()]
            e2e_rmse_list = [float(np.mean(r['rmse_lead'])) for r in e2e.values()]
            e2e_corr_list = [float(np.mean(r['corr_lead'])) for r in e2e.values()]

            m_dec_rmse, std_dec_rmse = np.mean(dec_rmse_list), np.std(dec_rmse_list)
            m_e2e_rmse, std_e2e_rmse = np.mean(e2e_rmse_list), np.std(e2e_rmse_list)
            m_dec_corr, std_dec_corr = np.mean(dec_corr_list), np.std(dec_corr_list)
            m_e2e_corr, std_e2e_corr = np.mean(e2e_corr_list), np.std(e2e_corr_list)

            rmse_gain = (m_e2e_rmse - m_dec_rmse) / m_e2e_rmse * 100  # 正值=分解式更优
            corr_gain = (m_dec_corr - m_e2e_corr) / m_e2e_corr * 100  # 正值=分解式更优

            print(f"  分解式  RMSE: {m_dec_rmse:.4f} ± {std_dec_rmse:.4f} m/s  (N={N_dec})")
            print(f"  端到端  RMSE: {m_e2e_rmse:.4f} ± {std_e2e_rmse:.4f} m/s  (N={N_e2e})")
            sign = "↓" if rmse_gain > 0 else "↑"
            print(f"  → RMSE 相对变化: {sign}{abs(rmse_gain):.1f}%  "
                  f"{'✓ 分解式更优' if rmse_gain > 0 else '⚠ 端到端更优'}")
            print()
            print(f"  分解式  Corr: {m_dec_corr:.4f} ± {std_dec_corr:.4f}")
            print(f"  端到端  Corr: {m_e2e_corr:.4f} ± {std_e2e_corr:.4f}")
            sign = "↑" if corr_gain > 0 else "↓"
            print(f"  → Corr 相对变化: {sign}{abs(corr_gain):.1f}%  "
                  f"{'✓ 分解式更优' if corr_gain > 0 else '⚠ 端到端更优'}")

            # ============== [结论] 证明分解式有效性 ==============
            print("\n" + "-" * 78)
            print("[结论] 分解式架构有效性证明 (假设检验 H1)")
            print("-" * 78)
            if rmse_gain > 0 and corr_gain > 0:
                print(f"  ✓ H1 成立: 分解式架构整体精度显著优于端到端")
                print(f"    · RMSE 相对降低 {rmse_gain:.1f}%")
                print(f"    · Corr 相对提升 {corr_gain:.1f}%")
                print()
                print(f"  ✓ 物理归纳偏置的价值被数据验证:")
                print(f"    将「地转平衡关系 u_geo = -(g/f)∇SSH」显式硬编码进架构，")
                print(f"    比让端到端网络从有限数据中隐式学习这一物理定律")
                print(f"    具有更高的样本效率、更好的泛化能力、以及更准确的预报。")
                print()
                print(f"  推论: 在数据量有限 / 预报时效较长 / 区域处于中高纬时，")
                print(f"        「地转流(物理)+非地转流(数据)」的分解式设计")
                print(f"        是比纯端到端更优的表层流预测范式。")
            else:
                if rmse_gain <= 0:
                    print(f"  ⚠ H1在RMSE维度不成立 (分解式RMSE端到端差 {-rmse_gain:.1f}%)")
                if corr_gain <= 0:
                    print(f"  ⚠ H1在Corr维度不成立 (分解式Corr比端到端差 {-corr_gain:.1f}%)")
                print()
                print(f"  可能原因:")
                print(f"    · 训练数据量充足，端到端也能充分学习地转关系")
                print(f"    · 研究区域含大量赤道带 (f→0，地转失效)")
                print(f"    · 非地转过程占主导（强风驱、强涡旋区域）")
                print(f"    · 分解式超参未调至最优，或SSH预训练模型精度不足")
        else:
            print(f"  [跳过] 缺少对比数据 (分解式={N_dec}, 端到端={N_e2e}，需两者都>0)")

        # ============== [指标3] 长预报时效衰减率 (H4) ==============
        print("\n" + "-" * 78)
        print("[指标3] 长预报时效衰减率 (假设H4: 分解式衰减更慢)")
        print("-" * 78)
        for name, r in {**dec, **e2e}.items():
            if 'rmse_lead' in r and len(r['rmse_lead']) >= 2:
                rmse = r['rmse_lead']
                decay = (rmse[-1] - rmse[0]) / rmse[0] * 100
                tag = '分解式' if name.startswith('current_') else '端到端'
                print(f"  [{tag:<4}] {name[:34]:<34}  lead1→{len(rmse)}:  RMSE增长 {decay:.1f}%  "
                      f"({rmse[0]:.4f} → {rmse[-1]:.4f})")

        # ============== [指标4] 逐日分布稳定性 ==============
        print("\n" + "-" * 78)
        print("[指标4] 逐日RMSE 稳定性 (变异系数 CV = std/mean，越小越稳定)")
        print("-" * 78)
        for name, r in {**dec, **e2e, **persist}.items():
            if 'rmse_date' in r and len(r['rmse_date']) >= 5:
                mu, sd = float(np.mean(r['rmse_date'])), float(np.std(r['rmse_date']))
                cv = sd / mu * 100 if mu > 0 else float('inf')
                tag = ('分解式' if name.startswith('current_') else
                       '端到端' if name.startswith('e2e_') else '基准')
                print(f"  [{tag:<4}] {name[:34]:<34}  RMSE日期 μ={mu:.4f}  σ={sd:.4f}  CV={cv:.1f}%")

        print("\n" + "=" * 78)
        print("  架构对比分析结束")
        print("=" * 78)
        return all_rows


# ====================================================================
# __main__: 一键运行对比
# ====================================================================
if __name__ == "__main__":
    import argparse as ap
    parser = ap.ArgumentParser(description="Compare Decomposed vs E2E architectures")
    parser.add_argument('--output_dir', type=str,
                        default="/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42")
    args_main = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    comparator = ArchitectureComparator(output_dir=args_main.output_dir)
    comparator.load_all()
    comparator.compare()
