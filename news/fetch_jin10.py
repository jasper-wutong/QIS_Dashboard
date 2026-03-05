#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金十数据快讯获取工具
"""

import requests
import json
import re
from datetime import datetime


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def get_latest_articles():
    """获取金十数据快讯"""
    timestamp = int(datetime.now().timestamp() * 1000)
    url = f"https://www.jin10.com/flash_newest.js?t={timestamp}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        raw_data = response.text
        json_str = re.sub(r'^var\s+newest\s*=\s*', '', raw_data)
        json_str = re.sub(r';*$', '', json_str).strip()
        
        data = json.loads(json_str)
        news_list = []
        
        for item in data:
            item_data = item.get("data", {})
            title = item_data.get("title") or item_data.get("content", "")
            
            if not title or (item.get("channel") and 5 in item.get("channel", [])):
                continue
            
            title = re.sub(r'<\/?b>', '', title)
            
            match = re.match(r'^【([^】]*)】(.*)$', title)
            if match:
                main_title = match.group(1)
                desc = match.group(2)
            else:
                main_title = title
                desc = ""
            
            news = {
                "id": item.get("id"),
                "title": main_title,
                "url": f"https://flash.jin10.com/detail/{item.get('id')}",
                "time": item.get("time", ""),
                "summary": desc,
                "important": bool(item.get("important"))
            }
            if news["title"]:
                news_list.append(news)
        
        return news_list
    except Exception as e:
        print(f"获取金十数据失败: {e}")
        return []


if __name__ == "__main__":
    news = get_latest_articles()
    for n in news[:5]:
        print(f"{n['title']}")
