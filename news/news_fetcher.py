#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻聚合器 - 从多个来源获取并分类新闻
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news import (
    fetch_10jqka,
    fetch_caixin,
    fetch_WSJ,
    fetch_eastmoney,
    fetch_jin10,
    fetch_wallstreetcn,
    fetch_36kr,
    fetch_baidu,
    fetch_zhihu,
    fetch_weibo,
    fetch_toutiao,
    fetch_thepaper,
    fetch_ithome,
    fetch_github,
    fetch_juejin,
    fetch_bilibili,
)


# 新闻源配置
NEWS_SOURCES = {
    # 财经类
    "finance": [
        {"name": "同花顺", "module": fetch_10jqka, "func": "get_live_news", "icon": "📈", "limit": 10},
        {"name": "财新网", "module": fetch_caixin, "func": "get_latest_articles", "icon": "📰", "limit": 10},
        {"name": "华尔街见闻", "module": fetch_WSJ, "func": "get_live_news", "icon": "💹", "limit": 10},
        {"name": "东方财富", "module": fetch_eastmoney, "func": "get_live_news", "icon": "💰", "limit": 10},
        {"name": "金十数据", "module": fetch_jin10, "func": "get_latest_articles", "icon": "⚡", "limit": 10},
        {"name": "华尔街见闻快讯", "module": fetch_wallstreetcn, "func": "get_latest_articles", "icon": "📊", "limit": 8},
        {"name": "36氪", "module": fetch_36kr, "func": "get_latest_articles", "icon": "🚀", "limit": 8},
    ],
    # 热搜类
    "hot": [
        {"name": "百度热搜", "module": fetch_baidu, "func": "get_latest_articles", "icon": "🔍", "limit": 10},
        {"name": "知乎热榜", "module": fetch_zhihu, "func": "get_latest_articles", "icon": "💬", "limit": 10},
        {"name": "微博热搜", "module": fetch_weibo, "func": "get_latest_articles", "icon": "🔥", "limit": 10},
        {"name": "今日头条", "module": fetch_toutiao, "func": "get_latest_articles", "icon": "📱", "limit": 10},
        {"name": "澎湃新闻", "module": fetch_thepaper, "func": "get_latest_articles", "icon": "📝", "limit": 8},
        {"name": "B站热搜", "module": fetch_bilibili, "func": "get_hot_search", "icon": "📺", "limit": 8},
    ],
    # 科技类
    "tech": [
        {"name": "IT之家", "module": fetch_ithome, "func": "get_latest_articles", "icon": "💻", "limit": 10},
        {"name": "GitHub Trending", "module": fetch_github, "func": "get_latest_articles", "icon": "🐙", "limit": 10},
        {"name": "掘金热榜", "module": fetch_juejin, "func": "get_latest_articles", "icon": "⛏️", "limit": 10},
    ],
}


def fetch_single_source(source_config):
    """从单个源获取新闻"""
    name = source_config["name"]
    module = source_config["module"]
    func_name = source_config["func"]
    icon = source_config["icon"]
    limit = source_config.get("limit", 10)
    
    try:
        func = getattr(module, func_name)
        news_list = func()
        
        # 添加源信息并限制数量
        result = []
        for news in news_list[:limit]:
            news["source"] = name
            news["source_icon"] = icon
            result.append(news)
        
        return {
            "name": name,
            "icon": icon,
            "news": result,
            "count": len(result),
            "success": True,
            "error": None
        }
    except Exception as e:
        print(f"获取 {name} 失败: {e}")
        return {
            "name": name,
            "icon": icon,
            "news": [],
            "count": 0,
            "success": False,
            "error": str(e)
        }


def fetch_all_news(max_workers=10, timeout=30):
    """
    并行获取所有新闻源
    
    返回格式:
    {
        "finance": [{"name": "同花顺", "icon": "📈", "news": [...], "count": 10, "success": True}, ...],
        "hot": [...],
        "tech": [...],
        "fetch_time": "2026-02-12 10:30:00",
        "total_count": 150,
        "success_count": 15,
        "failed_sources": ["彭博社"]
    }
    """
    result = {
        "finance": [],
        "hot": [],
        "tech": [],
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": 0,
        "success_count": 0,
        "failed_sources": []
    }
    
    # 收集所有任务
    all_tasks = []
    for category, sources in NEWS_SOURCES.items():
        for source in sources:
            all_tasks.append((category, source))
    
    # 并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(fetch_single_source, source): (category, source["name"])
            for category, source in all_tasks
        }
        
        for future in as_completed(future_to_task, timeout=timeout):
            category, source_name = future_to_task[future]
            try:
                source_result = future.result(timeout=5)
                result[category].append(source_result)
                result["total_count"] += source_result["count"]
                
                if source_result["success"]:
                    result["success_count"] += 1
                else:
                    result["failed_sources"].append(source_name)
            except Exception as e:
                print(f"获取 {source_name} 超时或出错: {e}")
                result["failed_sources"].append(source_name)
                result[category].append({
                    "name": source_name,
                    "icon": "❌",
                    "news": [],
                    "count": 0,
                    "success": False,
                    "error": str(e)
                })
    
    return result


def fetch_category_news(category, max_workers=5, timeout=20):
    """
    获取单个分类的新闻
    """
    if category not in NEWS_SOURCES:
        return {"error": f"未知分类: {category}", "news": []}
    
    sources = NEWS_SOURCES[category]
    result = {
        "category": category,
        "sources": [],
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": 0
    }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_source, s): s["name"] for s in sources}
        
        for future in as_completed(futures, timeout=timeout):
            try:
                source_result = future.result(timeout=5)
                result["sources"].append(source_result)
                result["total_count"] += source_result["count"]
            except Exception as e:
                source_name = futures[future]
                result["sources"].append({
                    "name": source_name,
                    "icon": "❌",
                    "news": [],
                    "count": 0,
                    "success": False,
                    "error": str(e)
                })
    
    return result


if __name__ == "__main__":
    print("🚀 开始获取所有新闻...")
    news_data = fetch_all_news()
    
    print(f"\n📊 获取结果统计:")
    print(f"   获取时间: {news_data['fetch_time']}")
    print(f"   总新闻数: {news_data['total_count']}")
    print(f"   成功源数: {news_data['success_count']}")
    print(f"   失败源: {news_data['failed_sources']}")
    
    for category in ["finance", "hot", "tech"]:
        print(f"\n📁 {category.upper()}:")
        for source in news_data[category]:
            status = "✅" if source["success"] else "❌"
            print(f"   {status} {source['icon']} {source['name']}: {source['count']}条")
