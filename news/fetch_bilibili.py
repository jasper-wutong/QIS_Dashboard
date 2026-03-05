#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站热搜和热门视频获取工具
"""

import requests
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def format_number(num):
    """格式化数字显示"""
    if num >= 10000:
        return f"{num // 10000}w+"
    return str(num)


def get_hot_search():
    """获取B站热搜"""
    url = "https://s.search.bilibili.com/main/hotword?limit=30"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for item in data.get("list", []):
            news = {
                "id": item.get("keyword"),
                "title": item.get("show_name", ""),
                "url": f"https://search.bilibili.com/all?keyword={requests.utils.quote(item.get('keyword', ''))}",
                "type": "search"
            }
            if news["title"]:
                news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取B站热搜失败: {e}")
        return []


def get_hot_videos():
    """获取B站热门视频"""
    url = "https://api.bilibili.com/x/web-interface/popular"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for video in data.get("data", {}).get("list", []):
            stat = video.get("stat", {})
            news = {
                "id": video.get("bvid"),
                "title": video.get("title", ""),
                "url": f"https://www.bilibili.com/video/{video.get('bvid')}",
                "owner": video.get("owner", {}).get("name", ""),
                "view": format_number(stat.get("view", 0)),
                "like": format_number(stat.get("like", 0)),
                "summary": video.get("desc", ""),
                "type": "video"
            }
            if news["title"]:
                news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取B站热门视频失败: {e}")
        return []


def get_latest_articles():
    """获取B站热搜（默认方法）"""
    return get_hot_search()


if __name__ == "__main__":
    news = get_hot_search()
    for n in news[:5]:
        print(f"{n['title']}")
