#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
东方财富新闻获取工具
"""

import requests
import json
import re
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.eastmoney.com/',
}


def get_live_news():
    """获取东方财富7x24小时快讯"""
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    params = {
        "cb": "jQuery_callback",
        "sr": -1,
        "page_size": 30,
        "page_index": 1,
        "ann_type": "SHA,SZA,BJA",
        "client_source": "web",
        "f_node": 0,
        "s_node": 0
    }
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        text = response.text
        match = re.search(r'jQuery_callback\((.*)\)', text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            data = response.json() if response.text.startswith('{') else {}
        
        news_list = []
        for item in data.get("data", {}).get("list", []):
            codes = item.get("codes", [{}])
            stock_name = codes[0].get("short_name", "") if codes else ""
            news = {
                "id": item.get("art_code"),
                "title": f"[{stock_name}] {item.get('title', '')}" if stock_name else item.get('title', ''),
                "url": f"https://data.eastmoney.com/notices/detail/{codes[0].get('stock_code', '') if codes else ''}/{item.get('art_code', '')}.html",
                "time": item.get("notice_date", ""),
                "source": stock_name
            }
            news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取快讯失败: {e}")
        return []


def get_latest_articles():
    """默认获取方法"""
    return get_live_news()


if __name__ == "__main__":
    news = get_live_news()
    for n in news[:5]:
        print(f"{n['title']}")
