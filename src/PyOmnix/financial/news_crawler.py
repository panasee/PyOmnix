import json
import os
from datetime import datetime

import akshare as ak
import pandas as pd

from pyomnix.consts import OMNIX_PATH
from pyomnix.omnix_logger import get_logger

from .utils.llm import call_llm
from .data_models import SentimentAnalysis

logger = get_logger("news_crawler")


def get_stock_news(symbol: str, max_news: int = 10) -> list:
    """获取并处理个股新闻

    Args:
        symbol (str): 股票代码，如 "300059"
        max_news (int, optional): 获取的新闻条数，默认为10条。最大支持100条。

    Returns:
        list: 新闻列表，每条新闻包含标题、内容、发布时间等信息
    """

    # 设置pandas显示选项，确保显示完整内容
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

    # 限制最大新闻条数
    max_news = min(max_news, 100)

    # 获取当前日期
    today = datetime.now().strftime("%Y-%m-%d")

    # 构建新闻文件路径
    news_dir = OMNIX_PATH / "financial" / "stock_news"
    print(f"新闻保存目录: {news_dir}")

    # 确保目录存在
    try:
        os.makedirs(news_dir, exist_ok=True)
        logger.debug(f"成功创建或确认目录存在: {news_dir}")
    except Exception as e:
        logger.error(f"创建目录失败: {e}")
        return []

    news_file = news_dir / f"{symbol}_news.json"
    logger.info(f"新闻文件路径: {news_file}")

    # 检查是否需要更新新闻
    need_update = True
    if news_file.exists():
        try:
            with open(news_file, encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == today:
                    cached_news = data.get("news", [])
                    if len(cached_news) >= max_news:
                        logger.info(f"使用缓存的新闻数据: {news_file}")
                        return cached_news[:max_news]
                    else:
                        logger.info(
                            f"缓存的新闻数量({len(cached_news)})不足，需要获取更多新闻({max_news}条)"
                        )
        except Exception as e:
            print(f"读取缓存文件失败: {e}")

    logger.info(f"开始获取{symbol}的新闻数据...")

    try:
        # 获取新闻列表
        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is None or len(news_df) == 0:
            logger.warning(f"未获取到{symbol}的新闻数据")
            return []

        logger.info(f"成功获取到{len(news_df)}条新闻")

        # 实际可获取的新闻数量
        available_news_count = len(news_df)
        if available_news_count < max_news:
            logger.warning(
                f"警告：实际可获取的新闻数量({available_news_count})少于请求的数量({max_news})"
            )
            max_news = available_news_count

        # 获取指定条数的新闻（考虑到可能有些新闻内容为空，多获取50%）
        news_list = []
        for _, row in news_df.head(int(max_news * 1.5)).iterrows():
            try:
                # 获取新闻内容
                content = (
                    row["新闻内容"]
                    if "新闻内容" in row and not pd.isna(row["新闻内容"])
                    else ""
                )
                if not content:
                    content = row["新闻标题"]

                # 只去除首尾空白字符
                content = content.strip()
                if len(content) < 10:  # 内容太短的跳过
                    continue

                # 获取关键词
                keyword = (
                    row["关键词"]
                    if "关键词" in row and not pd.isna(row["关键词"])
                    else ""
                )

                # 添加新闻
                news_item = {
                    "title": row["新闻标题"].strip(),
                    "content": content,
                    "publish_time": row["发布时间"],
                    "source": row["文章来源"].strip(),
                    "url": row["新闻链接"].strip(),
                    "keyword": keyword.strip(),
                }
                news_list.append(news_item)
                print(f"成功添加新闻: {news_item['title']}")

            except Exception as e:
                print(f"处理单条新闻时出错: {e}")
                continue

        # 按发布时间排序
        news_list.sort(key=lambda x: x["publish_time"], reverse=True)

        # 只保留指定条数的有效新闻
        news_list = news_list[:max_news]

        # 保存到文件
        try:
            save_data = {"date": today, "news": news_list}
            with open(news_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"成功保存{len(news_list)}条新闻到文件: {news_file}")
        except Exception as e:
            print(f"保存新闻数据到文件时出错: {e}")

        return news_list

    except Exception as e:
        print(f"获取新闻数据时出错: {e}")
        return []


def get_news_sentiment(news_list: list, num_of_news: int = 7) -> float:
    """分析新闻情感得分

    Args:
        news_list (list): 新闻列表
        num_of_news (int): 用于分析的新闻数量，默认为5条

    Returns:
        float: 情感得分，范围[-1, 1]，-1最消极，1最积极
    """
    if not news_list:
        return 0.0

    cache_file = OMNIX_PATH / "financial" / "sentiment_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # 生成新闻内容的唯一标识
    news_key = "|".join(
        [
            f"{news['title']}|{news['content'][:100]}|{news['publish_time']}"
            for news in news_list[:num_of_news]
        ]
    )

    # 检查缓存
    if cache_file.exists():
        print("发现情感分析缓存文件")
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
                if news_key in cache:
                    print("使用缓存的情感分析结果")
                    return cache[news_key]["sentiment_score"]
                print("未找到匹配的情感分析缓存")
        except Exception as e:
            print(f"读取情感分析缓存出错: {e}")
            cache = {}
    else:
        print("未找到情感分析缓存文件，将创建新文件")
        cache = {}

    # 准备系统消息
    system_message = {
        "role": "system",
        "content": """你是一个专业的A股市场分析师，擅长解读新闻对股票走势的影响。你需要：
        1. 根据预设评分标准分析新闻情感倾向
        2. 输出结构化分析结果（含评分、置信度、分析依据）

        评分标准速查：
        - 评分范围：[-1,1]，数值越大越积极
        - 示例参照：
          1.0 → 重大利好（如超预期财报）
          0.6 → 常规利好（如新订单）
          -0.8 → 严重利空（如财务造假）

        分析时重点关注：
        1. 业绩相关：财报、业绩预告、营收利润等
        2. 政策影响：行业政策、监管政策、地方政策等
        3. 市场表现：市场份额、竞争态势、商业模式等
        4. 资本运作：并购重组、股权激励、定增配股等
        5. 风险事件：诉讼仲裁、处罚、债务等
        6. 行业地位：技术创新、专利、市占率等
        7. 舆论环境：媒体评价、社会影响等

        请确保分析：
        1. 新闻的真实性和可靠性
        2. 新闻的时效性和影响范围
        3. 对公司基本面的实际影响
        4. A股市场的特殊反应规律""",
    }

    # 准备新闻内容
    news_content = "\n\n".join(
        [
            f"标题：{news['title']}\n"
            f"来源：{news['source']}\n"
            f"时间：{news['publish_time']}\n"
            f"内容：{news['content']}"
            for news in news_list[:num_of_news]  # 使用指定数量的新闻
        ]
    )

    user_message = {
        "role": "user",
        "content": f"请分析以下A股上市公司相关新闻的情感倾向：\n{news_content}",
    }

    try:
        # 获取LLM分析结果
        result = call_llm(
            prompt=user_message["content"],
            system_prompt=system_message["content"],
            model_name="deepseek-chat",  # Default to deepseek-chat for sentiment analysis
            provider_api="deepseek",  # Default to deepseek
            pydantic_model=SentimentAnalysis,
            agent_name="news_sentiment",
        )
        if result is None:
            print("Error: LLM API call failed, returned None")
            return 0.0

        sentiment_score = result.sentiment_score

        cache["sentiment_score"] = sentiment_score
        cache["confidence"] = result.confidence
        cache["reasoning"] = result.reasoning
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({news_key: cache}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing cache: {e}")

        return sentiment_score

    except Exception as e:
        print(f"Error analyzing news sentiment: {e}")
        return 0.0  # 出错时返回中性分数
