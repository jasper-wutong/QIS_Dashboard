#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财新网新闻获取工具
"""

import requests
import re
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.caixin.com/',
}


def get_latest_articles():
    """获取财新网最新文章"""
    url = "https://api.caixin.com/article/hotspot"
    params = {
        "channel": "finance",
        "limit": 30
    }
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_list = []
        for item in data.get("data", []):
            news = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "time": item.get("time", ""),
                "summary": item.get("summary", "")[:100] if item.get("summary") else ""
            }
            if news["title"]:
                news_list.append(news)
        
        return news_list
    except Exception as e:
        # 备用方案：从首页HTML抓取
        try:
            html_response = requests.get("https://www.caixin.com/", headers=HEADERS, timeout=10)
            html_response.raise_for_status()
            
            pattern = r'<a[^>]+href="(https?://[^"]*caixin\.com/[^"]*)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html_response.text)
            
            news_list = []
            seen_titles = set()
            for url, title in matches:
                title = title.strip()
                if title and len(title) > 5 and title not in seen_titles:
                    seen_titles.add(title)
                    news_list.append({
                        "title": title,
                        "url": url,
                        "time": "",
                        "summary": ""
                    })
            
            return news_list[:30]
        except Exception as e2:
            print(f"获取最新文章失败: {e}, 备用方案也失败: {e2}")
            return []


def get_breaking_news():
    """获取财新网金融新闻"""
    url = "https://finance.caixin.com/"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        pattern = r'<a[^>]+href="(https?://[^"]*caixin\.com/[^"]*\.html)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, response.text)
        
        news_list = []
        seen_titles = set()
        for url, title in matches:
            title = title.strip()
            if title and len(title) > 5 and title not in seen_titles:
                seen_titles.add(title)
                news_list.append({
                    "title": title,
                    "url": url,
                    "time": "",
                    "source": "财新金融"
                })
        
        return news_list[:30]
    except Exception as e:
        print(f"获取金融新闻失败: {e}")
        return []


if __name__ == "__main__":
    news = get_latest_articles()
    for n in news[:5]:
        print(f"{n['title']}")
