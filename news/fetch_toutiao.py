#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
今日头条热榜获取工具
"""

import requests
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def get_latest_articles():
    """获取今日头条热榜"""
    url = "https://www.toutiao.com/hot-event/hot-board/?origin=toutiao_pc"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for item in data.get("data", []):
            news = {
                "id": item.get("ClusterIdStr"),
                "title": item.get("Title", ""),
                "url": f"https://www.toutiao.com/trending/{item.get('ClusterIdStr')}/",
                "hot_value": item.get("HotValue", "")
            }
            if news["title"]:
                news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取今日头条热榜失败: {e}")
        return []


if __name__ == "__main__":
    news = get_latest_articles()
    for n in news[:5]:
        print(f"{n['title']}")
