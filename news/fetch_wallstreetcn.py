#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华尔街见闻快讯获取工具
"""

import requests
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def get_latest_articles():
    """获取华尔街见闻快讯"""
    url = "https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&limit=30"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for item in data.get("data", {}).get("items", []):
            news = {
                "id": item.get("id"),
                "title": item.get("title") or item.get("content_text", ""),
                "url": item.get("uri", ""),
                "time": datetime.fromtimestamp(item.get("display_time", 0)).strftime('%Y-%m-%d %H:%M:%S') if item.get("display_time") else ""
            }
            if news["title"]:
                news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取华尔街见闻失败: {e}")
        return []


if __name__ == "__main__":
    news = get_latest_articles()
    for n in news[:5]:
        print(f"{n['title']}")
