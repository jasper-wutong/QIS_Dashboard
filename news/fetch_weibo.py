#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微博热搜榜新闻获取工具
"""

import requests
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Cookie': 'SUB=_2AkMWIuNSf8NxqwJRmP8dy2rhaoV2ygrEieKgfhKJJRMxHRl-yT9jqk86tRB6PaLNvQZR6zYUcYVT1zSjoSreQHidcUq7',
}


def get_latest_articles():
    """获取微博热搜榜"""
    if not HAS_BS4:
        print("需要安装 beautifulsoup4: pip install beautifulsoup4")
        return []
    
    url = "https://s.weibo.com/top/summary?cate=realtimehot"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_list = []
        
        rows = soup.select('#pl_top_realtimehot table tbody tr')[1:]
        
        for row in rows:
            link = row.select_one('td.td-02 a')
            if not link or 'javascript:void(0)' in link.get('href', ''):
                continue
            
            title = link.text.strip()
            href = link.get('href', '')
            flag_text = row.select_one('td.td-03')
            flag = flag_text.text.strip() if flag_text else ""
            
            if title and href:
                news = {
                    "id": title,
                    "title": title,
                    "url": f"https://s.weibo.com{href}",
                    "flag": flag
                }
                news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取微博热搜失败: {e}")
        return []


if __name__ == "__main__":
    news = get_latest_articles()
    for n in news[:5]:
        print(f"{n['title']}")
