#!/usr/bin/env python3
"""
coordinator_agent.py - HeDA协调器Agent (Central Coordination Unit)
====================================================================

作为HeDA多Agent系统的中央协调单元，负责：
  1. 动态任务调度：按依赖关系编排三个子Agent的执行顺序
  2. 状态监控：实时跟踪各Agent和Stage的执行状态
  3. 错误恢复：支持从失败阶段重试或跳过
  4. 结果汇总：收集所有Agent的执行结果，生成最终报告

执行流程：
  ExtractionAgent (Stage 1-2)
    → KnowledgeArchitectAgent (Stage 3-5)
      → InferenceAgent (Stage 6-7)

对应论文架构中的 Central Coordination Unit (CCU)
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseAgent, ConfigManager, Stage, AgentState
from agents.extraction_agent import ExtractionAgent
from agents.knowledge_architect_agent import KnowledgeArchitectAgent
from agents.inference_agent import InferenceAgent


class CoordinatorAgent(BaseAgent):
    """协调器Agent - HeDA系统的中央协调单元

    管理三个子Agent的生命周期：
      - ExtractionAgent:          Stage 1 (语料获取) + Stage 2 (本体抽取)
      - KnowledgeArchitectAgent:  Stage 3 (语义消歧) + Stage 4 (属性增强) + Stage 5 (拓扑构建)
      - InferenceAgent:           Stage 6 (向量嵌入) + Stage 7 (推理验证)
    """

    def __init__(self):
        super().__init__(
            name="CoordinatorAgent",
            stages=list(Stage)  # 协调器监管所有7个阶段
        )
        self.sub_agents: Dict[str, BaseAgent] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.reports_dir = self.config.get('paths', {}).get('reports_dir', 'reports/')

    def _init_sub_agents(self):
        """初始化三个子Agent"""
        self.logger.info("初始化子Agent...")
        self.sub_agents = {
            'extraction': ExtractionAgent(),
            'knowledge_architect': KnowledgeArchitectAgent(),
            'inference': InferenceAgent(),
        }
        for name, agent in self.sub_agents.items():
            self.logger.info(f"  ✓ {agent.name} 就绪 (负责: {', '.join(s.name_cn for s in agent.stages)})")

    def _execute_agent(self, agent_key: str, skip_on_fail: bool = False) -> Dict[str, Any]:
        """执行单个子Agent并记录结果"""
        agent = self.sub_agents[agent_key]
        self.logger.info(f"\n{'━'*60}")
        self.logger.info(f"▶ 启动 {agent.name}")
        self.logger.info(f"{'━'*60}")

        start_time = time.time()
        try:
            result = agent.run()
        except Exception as e:
            result = {'success': False, 'message': f"Agent异常: {str(e)}"}

        elapsed = time.time() - start_time
        result['elapsed_seconds'] = round(elapsed, 2)

        log_entry = {
            'agent': agent.name,
            'agent_key': agent_key,
            'success': result.get('success', False),
            'elapsed_seconds': result['elapsed_seconds'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.execution_log.append(log_entry)

        status = "✅ 成功" if result.get('success') else "❌ 失败"
        self.logger.info(f"\n{status} {agent.name}(耗时: {elapsed:.1f}秒)")

        if not result.get('success') and not skip_on_fail:
            self.logger.error(f"  {agent.name} 执行失败，后续Agent将不会启动")

        return result

    def _generate_final_report(self, results: Dict[str, Any]) -> str:
        """生成最终执行报告"""
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        report = {
            'title': 'HeDA Pipeline Execution Report',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_success': results.get('success', False),
            'total_elapsed': sum(e.get('elapsed_seconds', 0) for e in self.execution_log),
            'agent_results': {},
            'execution_log': self.execution_log,
            'stage_summary': [],
        }

        # 各Agent结果摘要
        for key in ['extraction', 'knowledge_architect', 'inference']:
            agent_result = results.get('agent_results', {}).get(key, {})
            report['agent_results'][key] = {
                'success': agent_result.get('success', False),
                'elapsed': agent_result.get('elapsed_seconds', 0),
            }

        # 七阶段摘要
        for stage in Stage:
            report['stage_summary'].append({
                'stage_id': stage.stage_id,
                'name_en': stage.name_en,
                'name_cn': stage.name_cn,
            })

        report_file = Path(self.reports_dir) / 'pipeline_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"\n📄 执行报告已保存: {report_file}")
        return str(report_file)

    def run(self, skip_on_fail: bool = False,
            start_from: Optional[str] = None) -> Dict[str, Any]:
        """执行HeDA完整流水线

        Args:
            skip_on_fail: 某个Agent失败时是否继续执行后续Agent
            start_from: 从指定Agent开始执行 ('extraction'/'knowledge_architect'/'inference')
        """
        self.state = AgentState.RUNNING
        total_start = time.time()

        self.logger.info(f"{'╔'+'═'*58+'╗'}")
        self.logger.info(f"║{'HeDA Pipeline - Central Coordination Unit':^58}║")
        self.logger.info(f"║{'热浪灾害分析多Agent系统':^50}║")
        self.logger.info(f"{'╚'+'═'*58+'╝'}")

        # 初始化子Agent
        self._init_sub_agents()

        # 确定执行顺序
        agent_order = ['extraction', 'knowledge_architect', 'inference']
        if start_from and start_from in agent_order:
            start_idx = agent_order.index(start_from)
            agent_order = agent_order[start_idx:]
            self.logger.info(f"⏩ 从 {start_from}开始执行")

        agent_results = {}
        overall_success = True

        for agent_key in agent_order:
            result = self._execute_agent(agent_key, skip_on_fail)
            agent_results[agent_key] = result

            if not result.get('success'):
                overall_success = False
                if not skip_on_fail:
                    self.logger.error(f"流水线在 {agent_key}阶段终止")
                    break

        total_elapsed = time.time() - total_start
        self.state = AgentState.SUCCESS if overall_success else AgentState.FAILED

        # 最终汇总
        final_result = {
            'success': overall_success,
            'agent_results': agent_results,
            'total_elapsed_seconds': round(total_elapsed, 2),
        }

        # 生成报告
        report_file = self._generate_final_report(final_result)
        final_result['report_file'] = report_file

        # 打印总结
        self.logger.info(f"\n{'╔'+'═'*58+'╗'}")
        status = '✅ 全部成功' if overall_success else '❌ 存在失败'
        self.logger.info(f"║  执行结果: {status:48}║")
        self.logger.info(f"║  总耗时: {total_elapsed:.1f}秒{' '*(47-len(f'{total_elapsed:.1f}秒'))}║")
        self.logger.info(f"{'╚'+'═'*58+'╝'}")

        return final_result


    # ==============================================================
    #  便捷方法 - 单独运行某个子Agent
    # ==============================================================

    def run_extraction_only(self) -> Dict[str, Any]:
        """仅运行ExtractionAgent (Stage 1-2)"""
        self._init_sub_agents()
        return self._execute_agent('extraction')

    def run_knowledge_architect_only(self) -> Dict[str, Any]:
        """仅运行KnowledgeArchitectAgent (Stage 3-5)"""
        self._init_sub_agents()
        return self._execute_agent('knowledge_architect')

    def run_inference_only(self) -> Dict[str, Any]:
        """仅运行InferenceAgent (Stage 6-7)"""
        self._init_sub_agents()
        return self._execute_agent('inference')

    def get_status(self) -> Dict[str, Any]:
        """获取当前系统状态"""
        status = {
            'coordinator_state': self.state.value,
            'sub_agents': {},
            'execution_log': self.execution_log,
        }
        for key, agent in self.sub_agents.items():
            status['sub_agents'][key] = {
                'name': agent.name,
                'state': agent.state.value,
                'stages': [s.name_cn for s in agent.stages],
            }
        return status


# ============================================================
#  独立运行入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HeDA Pipeline Coordinator')
    parser.add_argument('--start-from', type=str, default=None,
                       choices=['extraction', 'knowledge_architect', 'inference'],
                       help='从指定Agent开始执行')
    parser.add_argument('--skip-on-fail', action='store_true',
                       help='某个Agent失败时继续执行后续Agent')
    parser.add_argument('--agent', type=str, default=None,
                       choices=['extraction', 'knowledge_architect', 'inference'],
                       help='仅运行指定的子Agent')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )

    coordinator = CoordinatorAgent()

    if args.agent:
        # 单独运行某个Agent
        method_map = {
            'extraction': coordinator.run_extraction_only,
            'knowledge_architect': coordinator.run_knowledge_architect_only,
            'inference': coordinator.run_inference_only,
        }
        result = method_map[args.agent]()
    else:
        # 运行完整流水线
        result = coordinator.run(
            skip_on_fail=args.skip_on_fail,
            start_from=args.start_from,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))