#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华尔街见闻新闻获取工具
"""

import requests
from datetime import datetime


def get_live_news():
    """获取华尔街见闻快讯（实时资讯）"""
    url = "https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&limit=30"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for item in data.get("data", {}).get("items", []):
            news = {
                "id": item.get("id"),
                "title": item.get("title") or item.get("content_text", ""),
                "url": item.get("uri", ""),
                "time": datetime.fromtimestamp(item.get("display_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if item.get("display_time") else "",
                "digest": item.get("content_text", "")
            }
            if news["title"] == news["digest"]:
                news["digest"] = ""
            news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取快讯失败: {e}")
        return []


def get_articles():
    """获取华尔街见闻文章（深度报道）"""
    url = "https://api-one.wallstcn.com/apiv1/content/information-flow?channel=global-channel&accept=article&limit=30"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for item in data.get("data", {}).get("items", []):
            resource_type = item.get("resource_type", "")
            resource = item.get("resource", {})
            
            if resource_type in ["theme", "ad"]:
                continue
            if resource.get("type") == "live":
                continue
            if not resource.get("uri"):
                continue
            
            news = {
                "id": resource.get("id"),
                "title": resource.get("title") or resource.get("content_short", ""),
                "url": resource.get("uri", ""),
                "time": datetime.fromtimestamp(resource.get("display_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if resource.get("display_time") else "",
                "digest": resource.get("content_short", "")
            }
            news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取文章失败: {e}")
        return []


def get_latest_articles():
    """默认获取方法"""
    return get_live_news()


if __name__ == "__main__":
    news = get_live_news()
    for n in news[:5]:
        print(f"{n['title']}")
