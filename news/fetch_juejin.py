#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掘金热榜获取工具
"""

import requests
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def get_latest_articles():
    """获取掘金热榜"""
    url = "https://api.juejin.cn/content_api/v1/content/article_rank?category_id=1&type=hot&spider=0"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for item in data.get("data", []):
            content = item.get("content", {})
            content_id = content.get("content_id", "")
            
            news = {
                "id": content_id,
                "title": content.get("title", ""),
                "url": f"https://juejin.cn/post/{content_id}"
            }
            if news["title"]:
                news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取掘金热榜失败: {e}")
        return []


if __name__ == "__main__":
    news = get_latest_articles()
    for n in news[:5]:
        print(f"{n['title']}")
