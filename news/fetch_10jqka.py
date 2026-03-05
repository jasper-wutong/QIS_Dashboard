#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同花顺新闻获取工具
"""

import requests
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.10jqka.com.cn/',
}


def get_live_news():
    """获取同花顺7x24小时快讯"""
    url = "https://news.10jqka.com.cn/tapp/news/push/stock/"
    params = {
        "page": 1,
        "tag": "",
        "track": "website",
        "pagesize": 30
    }
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        raw_list = data.get("data", {}).get("list", [])
        
        for item in raw_list:
            # 解析时间
            time_str = ""
            ctime = item.get("ctime", 0)
            if ctime:
                try:
                    time_str = datetime.fromtimestamp(int(ctime)).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = str(ctime)
            
            news = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "url": item.get("url", "") or f"https://news.10jqka.com.cn/{item.get('id')}/",
                "time": time_str,
                "source": item.get("source", ""),
                "digest": item.get("digest", "")
            }
            news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取快讯失败: {e}")
        return []


def get_important_news():
    """获取同花顺要闻精选"""
    url = "https://news.10jqka.com.cn/tapp/news/push/stock/?page=1&tag=-20000&track=website&pagesize=30"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        response_data = data.get("data")
        if not response_data or not isinstance(response_data, dict):
            return []
            
        news_list = []
        for item in response_data.get("list", []):
            time_str = ""
            ctime = item.get("ctime", 0)
            if ctime:
                try:
                    time_str = datetime.fromtimestamp(int(ctime)).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = str(ctime)
            
            news = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "time": time_str,
                "digest": item.get("digest", "")[:100] if item.get("digest") else ""
            }
            news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取要闻精选失败: {e}")
        return []


def get_latest_articles():
    """默认获取方法"""
    return get_live_news()


if __name__ == "__main__":
    news = get_live_news()
    for n in news[:5]:
        print(f"{n['title']}")
