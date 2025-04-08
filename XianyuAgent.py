from http import HTTPStatus
from dashscope import Application
import re
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI
from loguru import logger
import json
import time
import requests
import logging

# 配置日志
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# 功能配置
INCLUDE_HISTORY = os.getenv('INCLUDE_HISTORY', 'true').lower() == 'true'
INCLUDE_PRODUCT_INFO = os.getenv('INCLUDE_PRODUCT_INFO', 'true').lower() == 'true'

class XianyuReplyBot:
    def __init__(self):
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.dashscope_app_id = os.getenv('DASHSCOPE_APP_ID')
        if not self.dashscope_app_id:
            logger.warning("DASHSCOPE_APP_ID not set. Product information feature will be disabled.")
            INCLUDE_PRODUCT_INFO = False
        self._init_system_prompts()
        self._load_agent_config()  # 加载专家配置
        self._init_agents()
        self.router = IntentRouter(self.agents['classify'])
        self.last_intent = None  # 记录最后一次意图

    def _load_agent_config(self):
        """加载专家配置"""
        # 默认启用所有专家
        self.enabled_agents = {
            'classify': True,  # 分类专家始终启用
            'price': True,
            'tech': True,
            'product': True,
            'default': True
        }
        
        # 从环境变量加载配置
        agent_config = os.getenv("ENABLED_AGENTS", "")
        if agent_config:
            try:
                # 解析配置，格式为 "agent1,agent2,agent3"
                enabled_list = [agent.strip() for agent in agent_config.split(",")]
                # 重置所有专家为禁用状态
                for agent in self.enabled_agents:
                    self.enabled_agents[agent] = False
                # 设置启用的专家
                for agent in enabled_list:
                    if agent in self.enabled_agents:
                        self.enabled_agents[agent] = True
                
                # 确保至少有一个专家被启用
                if not any(self.enabled_agents.values()):
                    logger.warning("没有启用任何专家，将启用默认专家")
                    self.enabled_agents['default'] = True
                
                logger.info(f"已加载专家配置: {self.enabled_agents}")
            except Exception as e:
                logger.error(f"加载专家配置出错: {e}")
                # 出错时启用所有专家
                for agent in self.enabled_agents:
                    self.enabled_agents[agent] = True
        
        # 加载其他配置
        self.include_history = INCLUDE_HISTORY
        self.include_product_info = INCLUDE_PRODUCT_INFO
        
        logger.info(f"配置加载完成: 包含历史记录={self.include_history}, 包含产品信息={self.include_product_info}")

    def _init_agents(self):
        """初始化各领域Agent"""
        self.agents = {}
        
        # 始终初始化分类专家，因为路由需要
        self.agents['classify'] = ClassifyAgent(self.client, self.classify_prompt, self._safe_filter)
        
        # 根据配置初始化其他专家
        if self.enabled_agents.get('price', True):
            self.agents['price'] = PriceAgent(self.client, self.price_prompt, self._safe_filter)
        
        if self.enabled_agents.get('tech', True):
            self.agents['tech'] = TechAgent(self.client, self.tech_prompt, self._safe_filter)
        
        if self.enabled_agents.get('product', True):
            self.agents['product'] = ProductAgent(self.product_prompt, self._safe_filter)
        
        if self.enabled_agents.get('default', True):
            self.agents['default'] = DefaultAgent(self.client, self.default_prompt, self._safe_filter)
        
        logger.info(f"已初始化专家: {list(self.agents.keys())}")

    def _init_system_prompts(self):
        """初始化各Agent专用提示词，直接从文件中加载"""
        prompt_dir = "prompts"
        
        try:
            # 加载分类提示词
            with open(os.path.join(prompt_dir, "classify_prompt.txt"), "r", encoding="utf-8") as f:
                self.classify_prompt = f.read()
                logger.debug(f"已加载分类提示词，长度: {len(self.classify_prompt)} 字符")
            
            # 加载价格提示词
            with open(os.path.join(prompt_dir, "price_prompt.txt"), "r", encoding="utf-8") as f:
                self.price_prompt = f.read()
                logger.debug(f"已加载价格提示词，长度: {len(self.price_prompt)} 字符")
            
            # 加载技术提示词
            with open(os.path.join(prompt_dir, "tech_prompt.txt"), "r", encoding="utf-8") as f:
                self.tech_prompt = f.read()
                logger.debug(f"已加载技术提示词，长度: {len(self.tech_prompt)} 字符")
            
            # 加载产品提示词
            with open(os.path.join(prompt_dir, "product_prompt.txt"), "r", encoding="utf-8") as f:
                self.product_prompt = f.read()
                logger.debug(f"已加载产品提示词，长度: {len(self.product_prompt)} 字符")
            
            # 加载默认提示词
            with open(os.path.join(prompt_dir, "default_prompt.txt"), "r", encoding="utf-8") as f:
                self.default_prompt = f.read()
                logger.debug(f"已加载默认提示词，长度: {len(self.default_prompt)} 字符")
                
            logger.info("成功加载所有提示词")
        except Exception as e:
            logger.error(f"加载提示词时出错: {e}")
            raise

    def _safe_filter(self, text: str) -> str:
        """安全过滤模块"""
        return text
        #blocked_phrases = ["微信", "QQ", "支付宝", "银行卡", "线下"]
        #return "[安全提醒]请通过平台沟通" if any(p in text for p in blocked_phrases) else text

    def format_history(self, context: List[Dict]) -> str:
        """格式化对话历史，返回完整的对话记录"""
        if not self.include_history:
            return ""
            
        # 过滤掉系统消息，只保留用户和助手的对话
        user_assistant_msgs = [msg for msg in context if msg['role'] in ['user', 'assistant']]
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in user_assistant_msgs])

    def generate_reply(self, user_msg: str, item_desc: str, context: List[Dict]) -> str:
        """生成回复主流程"""
        # 记录用户消息
        # logger.debug(f'用户所发消息: {user_msg}')
        
        formatted_context = self.format_history(context)
        # logger.debug(f'对话历史: {formatted_context}')
        
        # 检查是否有多个专家可用（不包括分类专家）
        available_agents = [agent for agent in self.agents.keys() if agent != 'classify']
        
        if len(available_agents) == 1:
            # 如果只有一个专家可用，直接使用它，跳过路由
            agent = self.agents[available_agents[0]]
            logger.info(f'只有一个专家可用，直接使用: {available_agents[0]}')
            self.last_intent = available_agents[0]
        else:
            # 如果有多个专家，使用路由决策
            # 1. 路由决策
            detected_intent = self.router.detect(user_msg, item_desc, formatted_context)
            
            # 2. 获取对应Agent
            internal_intents = {'classify'}  # 定义不对外开放的Agent
            
            if detected_intent in self.agents and detected_intent not in internal_intents:
                # 使用路由结果
                agent = self.agents[detected_intent]
                logger.info(f'意图识别完成: {detected_intent}')
                self.last_intent = detected_intent
            else:
                # 如果路由结果不可用，使用默认专家
                agent = self.agents['default']
                logger.info(f'意图识别完成: default')
                self.last_intent = 'default'
        
        # 3. 获取议价次数
        bargain_count = self._extract_bargain_count(context)
        logger.info(f'议价次数: {bargain_count}')

        # 4. 生成回复
        return agent.generate(
            user_msg=user_msg,
            item_desc=item_desc,
            context=formatted_context,
            bargain_count=bargain_count
        )
    
    def _extract_bargain_count(self, context: List[Dict]) -> int:
        """
        从上下文中提取议价次数信息
        
        Args:
            context: 对话历史
            
        Returns:
            int: 议价次数，如果没有找到则返回0
        """
        # 查找系统消息中的议价次数信息
        for msg in context:
            if msg['role'] == 'system' and '议价次数' in msg['content']:
                try:
                    # 提取议价次数
                    match = re.search(r'议价次数[:：]\s*(\d+)', msg['content'])
                    if match:
                        return int(match.group(1))
                except Exception:
                    pass
        return 0

    def reload_prompts(self):
        """重新加载所有提示词"""
        logger.info("正在重新加载提示词...")
        self._init_system_prompts()
        self._init_agents()
        logger.info("提示词重新加载完成")


class IntentRouter:
    """意图路由决策器"""

    def __init__(self, classify_agent):
        self.rules = {
            'tech': {  # 技术类优先判定
                'keywords': ['参数', '规格', '型号', '连接', '对比'],
                'patterns': [
                    r'和.+比'             
                ]
            },
            'price': {
                'keywords': ['便宜', '价', '砍价', '少点'],
                'patterns': [r'\d+元', r'能少\d+']
            },
            'product': {  # 添加产品类规则
                'keywords': ['产品', '商品', '功能', '特点', '优点', '缺点', '使用', '怎么用', '怎么样'],
                'patterns': [r'这个.+怎么样', r'这个.+怎么用']
            }
        }
        self.classify_agent = classify_agent

    def detect(self, user_msg: str, item_desc, context) -> str:
        """三级路由策略（技术优先）"""
        text_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', user_msg)
        
        # 1. 技术类关键词优先检查
        if any(kw in text_clean for kw in self.rules['tech']['keywords']):
            # logger.debug(f"技术类关键词匹配: {[kw for kw in self.rules['tech']['keywords'] if kw in text_clean]}")
            return 'tech'
            
        # 2. 技术类正则优先检查
        for pattern in self.rules['tech']['patterns']:
            if re.search(pattern, text_clean):
                # logger.debug(f"技术类正则匹配: {pattern}")
                return 'tech'

        # 3. 价格类检查
        for intent in ['price']:
            if any(kw in text_clean for kw in self.rules[intent]['keywords']):
                # logger.debug(f"价格类关键词匹配: {[kw for kw in self.rules[intent]['keywords'] if kw in text_clean]}")
                return intent
            
            for pattern in self.rules[intent]['patterns']:
                if re.search(pattern, text_clean):
                    # logger.debug(f"价格类正则匹配: {pattern}")
                    return intent
        
        # 4. 产品类检查
        for intent in ['product']:
            if any(kw in text_clean for kw in self.rules[intent]['keywords']):
                logger.debug(f"产品类关键词匹配: {[kw for kw in self.rules[intent]['keywords'] if kw in text_clean]}")
                return intent
            
            for pattern in self.rules[intent]['patterns']:
                if re.search(pattern, text_clean):
                    logger.debug(f"产品类正则匹配: {pattern}")
                    return intent
        
        # 5. 大模型兜底
        # logger.debug("使用大模型进行意图分类")
        return self.classify_agent.generate(
            user_msg=user_msg,
            item_desc=item_desc,
            context=context
        )


class BaseAgent:
    """Agent基类"""

    def __init__(self, client, system_prompt, safety_filter):
        self.client = client
        self.system_prompt = system_prompt
        self.safety_filter = safety_filter
        self.include_product_info = INCLUDE_PRODUCT_INFO

    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int = 0) -> str:
        """生成回复模板方法"""
        messages = self._build_messages(user_msg, item_desc, context)
        response = self._call_llm(messages)
        return self.safety_filter(response)

    def _build_messages(self, user_msg: str, item_desc: str, context: str) -> List[Dict]:
        """构建消息链"""
        system_content = ""
        
        if self.include_product_info:
            system_content += f"【商品信息】{item_desc}\n"
        
        if context:
            system_content += f"【你与客户对话历史】{context}\n"
            
        system_content += self.system_prompt
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_msg}
        ]

    def _call_llm(self, messages: List[Dict], temperature: float = 0.4) -> str:
        """调用大模型"""
        response = self.client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            temperature=temperature,
            max_tokens=500,
            top_p=0.8
        )
        return response.choices[0].message.content


class PriceAgent(BaseAgent):
    """议价处理Agent"""

    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int=0) -> str:
        """重写生成逻辑"""
        dynamic_temp = self._calc_temperature(bargain_count)
        messages = self._build_messages(user_msg, item_desc, context)
        messages[0]['content'] += f"\n▲当前议价轮次：{bargain_count}"

        response = self.client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            temperature=dynamic_temp,
            max_tokens=500,
            top_p=0.8
        )
        return self.safety_filter(response.choices[0].message.content)

    def _calc_temperature(self, bargain_count: int) -> float:
        """动态温度策略"""
        return min(0.3 + bargain_count * 0.15, 0.9)


class TechAgent(BaseAgent):
    """技术咨询Agent"""
    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int=0) -> str:
        """重写生成逻辑"""
        messages = self._build_messages(user_msg, item_desc, context)
        # messages[0]['content'] += "\n▲知识库：\n" + self._fetch_tech_specs()

        response = self.client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            temperature=0.4,
            max_tokens=500,
            top_p=0.8,
            extra_body={
                "enable_search": True,
            }
        )

        return self.safety_filter(response.choices[0].message.content)


    # def _fetch_tech_specs(self) -> str:
    #     """模拟获取技术参数（可连接数据库）"""
    #     return "功率：200W@8Ω\n接口：XLR+RCA\n频响：20Hz-20kHz"


class ClassifyAgent(BaseAgent):
    """意图识别Agent"""

    def generate(self, **args) -> str:
        response = super().generate(**args)
        return response


class DefaultAgent(BaseAgent):
    """默认处理Agent"""

    def _call_llm(self, messages: List[Dict], *args) -> str:
        """限制默认回复长度"""
        response = super()._call_llm(messages, temperature=0.7)
        return response


class ProductAgent:
    """产品信息咨询Agent"""
    def __init__(self, system_prompt, safety_filter):
        self.system_prompt = system_prompt
        self.safety_filter = safety_filter
        self.api_key = os.getenv("OPENAI_API_KEY")  # 使用 OPENAI_API_KEY 而不是 DASHSCOPE_API_KEY
        self.app_id = os.getenv("DASHSCOPE_APP_ID")  # 从环境变量获取应用ID
        # 从环境变量获取日志级别，默认为 INFO
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        # 是否打印详细日志
        self.verbose_logging = self.log_level in ["DEBUG", "TRACE"]
        # 是否包含历史记录
        self.include_history = INCLUDE_HISTORY
        # 是否包含产品信息
        self.include_product_info = INCLUDE_PRODUCT_INFO
        
        logger.info(f"初始化产品专家，日志级别: {self.log_level}, 包含历史记录: {self.include_history}, 包含产品信息: {self.include_product_info}")

    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int=0) -> str:
        """生成回复"""
        try:
            # 构建消息列表
            system_content = self.system_prompt
            
            if self.include_product_info:
                system_content += f"\n\n商品信息：{item_desc}"
                
            messages = [
                {
                    "role": "system",
                    "content": system_content
                }
            ]
            
            # 解析上下文历史
            if self.include_history and context:
                history_lines = context.strip().split('\n')
                for line in history_lines:
                    if line.startswith('user:'):
                        messages.append({"role": "user", "content": line[5:].strip()})
                    elif line.startswith('assistant:'):
                        messages.append({"role": "assistant", "content": line[10:].strip()})
            
            # 添加当前用户问题
            messages.append({"role": "user", "content": user_msg})
            
            # 记录请求信息
            if self.verbose_logging:
                logger.debug("【产品信息专家】请求信息:")
                logger.debug(f"App ID: {self.app_id}")
                logger.debug(f"消息列表: {json.dumps(messages, ensure_ascii=False, indent=2)}")
                logger.debug(f"温度: 0.7, Top P: 0.8")
            
            # 调用API
            start_time = time.time()
            
            # 构建完整的提示词
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"系统: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"用户: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"助手: {msg['content']}\n"
            
            logger.info(f"【产品信息专家】调用API，提示词长度: {len(prompt)}")
            
            # 调用API
            response = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=prompt,
                temperature=0.7,
                top_p=0.8,
                stream=False
            )
            
            end_time = time.time()
            
            # 记录响应信息
            if self.verbose_logging:
                logger.debug("【产品信息专家】响应信息:")
                logger.debug(f"请求ID: {response.request_id}")
                logger.debug(f"状态码: {response.status_code}")
                logger.debug(f"响应时间: {end_time - start_time:.2f}秒")
                logger.debug(f"原始响应: {response.output.text}")
            
            if response.status_code != HTTPStatus.OK:
                logger.error(f"产品信息API调用失败: request_id={response.request_id}, code={response.status_code}")
                return "抱歉，我现在无法回答这个问题，请稍后再试。"
            
            # 应用安全过滤
            filtered_response = self.safety_filter(response.output.text)
            
            # 记录过滤后的响应
            if self.verbose_logging:
                logger.debug("【产品信息专家】过滤后响应:")
                logger.debug(filtered_response)
            
            logger.info(f"【产品信息专家】成功生成回复，长度: {len(filtered_response)}")
            return filtered_response
            
        except Exception as e:
            logger.error(f"产品信息API调用异常: {e}")
            if self.verbose_logging:
                logger.exception("详细异常信息:")
            return "抱歉，系统出现了一些问题，请稍后再试。"