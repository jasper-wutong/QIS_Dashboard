"""
下载期货研报（概要版）
从Coremail邮件系统下载期货观点邮件，并将邮件中的链接页面保存为PDF

使用方法:
1. 复制 param.json.example 为 param.json 并填入邮箱账号密码
2. 安装依赖: pip install selenium requests
3. 运行: python download_futures_reports.py

可配置项:
- SAVE_DIR: PDF保存目录
- EMAIL_FOLDER: 邮件文件夹名称
- EMAIL_PATTERN: 邮件主题搜索关键字
- EMAIL_LIMIT: 处理邮件数量限制
"""

import os
import sys

# 设置输出编码为 UTF-8，避免 Windows 批处理中的 GBK 编码错误
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# 保存原始代理设置供 Chrome 使用
_saved_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

# 确保只绕过 localhost 和内部服务器的代理
os.environ['NO_PROXY'] = '127.0.0.1,localhost,*.cicc.group,mailbj.cicc.group'
os.environ['no_proxy'] = '127.0.0.1,localhost,*.cicc.group,mailbj.cicc.group'

# Patch Selenium 远程连接使用无代理连接
import urllib3
urllib3.disable_warnings()

from coremail_helper import CoremailHelper
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote import remote_connection as rc

# Patch Selenium 的 RemoteConnection 来使用无代理连接 ChromeDriver
_orig_init = rc.RemoteConnection.__init__
def _patched_init(self, *args, **kwargs):
    kwargs['ignore_proxy'] = True
    _orig_init(self, *args, **kwargs)
rc.RemoteConnection.__init__ = _patched_init

import time
import re
import glob as glob_module
from datetime import datetime

# ============ 配置区域 ============
# 保存目录（可根据需要修改）
BASE_SAVE_DIR = r'D:\QIS_DASHBOARD\memory\commodity_general_reports'

# 邮件搜索配置
EMAIL_FOLDER = '收件箱.期货研报'  # 邮件文件夹
EMAIL_PATTERN = '订阅每日期货观点'  # 搜索关键字
EMAIL_SEARCH_LIMIT = 200  # 搜索邮件数量上限（设大一点确保覆盖）

# 目标日期配置（只下载指定日期的报告）
# 设为 None 表示下载当天的报告
# 也可以设为具体日期，如 '2026-02-10'
TARGET_DATE = None
# =================================

# 初始化 Coremail Helper（从 param.json 读取配置）
helper = CoremailHelper()

# 确保基础保存目录存在
if not os.path.exists(BASE_SAVE_DIR):
    os.makedirs(BASE_SAVE_DIR)
    print(f'创建目录: {BASE_SAVE_DIR}')


def get_date_subfolder(email_date_str):
    """
    根据邮件接收日期创建子文件夹路径
    
    Args:
        email_date_str: 邮件接收日期字符串（格式如 '2026-02-10' 或 ISO格式）
    
    Returns:
        子文件夹的完整路径
    """
    # 解析日期字符串，只取日期部分
    if 'T' in email_date_str:
        # ISO格式: 2026-02-10T08:30:00+08:00
        date_part = email_date_str.split('T')[0]
    else:
        date_part = email_date_str[:10]
    
    subfolder_name = f'{date_part} general reports'
    subfolder_path = os.path.join(BASE_SAVE_DIR, subfolder_name)
    
    # 创建子文件夹（如果不存在）
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        print(f'创建日期子文件夹: {subfolder_path}')
    
    return subfolder_path


def get_existing_report_ids():
    """扫描所有日期子文件夹中已下载的PDF文件，提取已有的 taskId_id 组合，用于去重"""
    existing = set()
    if os.path.exists(BASE_SAVE_DIR):
        # 扫描所有子文件夹（日期文件夹）
        for subfolder in os.listdir(BASE_SAVE_DIR):
            subfolder_path = os.path.join(BASE_SAVE_DIR, subfolder)
            if os.path.isdir(subfolder_path):
                for f in glob_module.glob(os.path.join(subfolder_path, '期货观点_*.pdf')):
                    basename = os.path.basename(f)
                    # 文件名格式: 期货观点_{taskId}_{id}_{title}.pdf
                    match = re.match(r'期货观点_(\d+_\d+)', basename)
                    if match:
                        existing.add(match.group(1))
        # 也扫描根目录（兼容旧文件）
        for f in glob_module.glob(os.path.join(BASE_SAVE_DIR, '期货观点_*.pdf')):
            basename = os.path.basename(f)
            match = re.match(r'期货观点_(\d+_\d+)', basename)
            if match:
                existing.add(match.group(1))
    return existing

# 设置Chrome选项
chrome_options = Options()
chrome_options.add_argument('--headless=new')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--window-size=1920,1080')
chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
# 设置 Chrome 代理（用于访问外部网站）
if _saved_proxy:
    chrome_options.add_argument(f'--proxy-server={_saved_proxy}')


def extract_link_from_email(email_id):
    """从邮件中提取链接"""
    try:
        html_content = helper._CoremailHelper__handler.getHTML(email_id)
        if html_content:
            pattern = r'href=["\']([^"\']*taskId[^"\']*)["\']'
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                return matches[0]
    except Exception as e:
        print(f'提取链接失败: {str(e)}')
    return None


def save_page_as_pdf(driver, url, filename, save_dir, report_date=None):
    """使用Selenium访问页面并保存为PDF
    
    Args:
        driver: Selenium WebDriver实例
        url: 要访问的URL
        filename: 文件名前缀
        save_dir: 保存目录
        report_date: 报告日期（用于追加到文件名）
    """
    try:
        print(f'  正在访问: {url[:80]}...')
        driver.get(url)
        time.sleep(5)
        
        print(f'  正在加载完整页面内容...')
        
        # 找到实际的可滚动内容容器（scrollHeight > clientHeight）
        scroll_result = driver.execute_script("""
            var elements = Array.from(document.querySelectorAll('*'));
            var scrollable = elements.filter(el => el.scrollHeight > el.clientHeight && el.scrollHeight > 500);
            scrollable.sort((a, b) => b.scrollHeight - a.scrollHeight);
            if (scrollable.length > 0) {
                return {
                    found: true,
                    scrollHeight: scrollable[0].scrollHeight,
                    clientHeight: scrollable[0].clientHeight
                };
            }
            return {found: false};
        """)
        
        if scroll_result['found']:
            print(f'  找到可滚动容器: scrollHeight={scroll_result["scrollHeight"]}px, clientHeight={scroll_result["clientHeight"]}px')
            content_height = scroll_result['scrollHeight']
            
            # 找到并滚动这个容器
            scroll_container = driver.execute_script("""
                var elements = Array.from(document.querySelectorAll('*'));
                var scrollable = elements.filter(el => el.scrollHeight > el.clientHeight && el.scrollHeight > 500);
                scrollable.sort((a, b) => b.scrollHeight - a.scrollHeight);
                return scrollable[0];
            """)
            
            # 滚动加载所有内容
            scroll_attempts = 0
            max_scrolls = 20
            
            while scroll_attempts < max_scrolls:
                last_scroll = driver.execute_script("return arguments[0].scrollTop", scroll_container)
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_container)
                time.sleep(2)
                new_scroll = driver.execute_script("return arguments[0].scrollTop", scroll_container)
                
                if new_scroll == last_scroll:
                    break
                scroll_attempts += 1
                print(f'  滚动进度: {scroll_attempts}/{max_scrolls}')
            
            # 滚动回顶部
            driver.execute_script("arguments[0].scrollTop = 0", scroll_container)
            time.sleep(1)
            print(f'  滚动完成（{scroll_attempts + 1}次）')
        else:
            content_height = 1000
            print(f'  未找到可滚动容器，使用默认高度')

        # 获取页面标题
        page_title = driver.title
        print(f'  页面标题: {page_title}')
        
        # 尝试从页面中提取更完整的标题（包含日期）
        full_title = None
        try:
            # 尝试查找包含"研报观点"的标题元素
            full_title = driver.execute_script("""
                var elements = document.querySelectorAll('h1, h2, .title, [class*="title"]');
                for (var el of elements) {
                    var text = el.innerText || el.textContent;
                    if (text && (text.includes('期货') || text.includes('研报')) && /\\d{4}-\\d{2}-\\d{2}/.test(text)) {
                        return text.trim();
                    }
                }
                return null;
            """)
            if full_title:
                print(f'  完整标题: {full_title}')
        except:
            pass
        
        # 使用完整标题（如果找到）或页面标题
        title_to_use = full_title if full_title else page_title
        
        # 清理文件名
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', title_to_use) if title_to_use else ''
        
        # 如果标题中没有日期，且提供了report_date，则追加日期
        if safe_title and report_date:
            if not re.search(r'\d{4}-\d{2}-\d{2}', safe_title):
                safe_title = f'{safe_title} {report_date}'
                print(f'  追加日期到文件名: {report_date}')
        
        if safe_title:
            filename = f'{filename}_{safe_title}.pdf'
        else:
            filename = f'{filename}.pdf'
        
        filepath = os.path.join(save_dir, filename)
        
        # 计算PDF高度（使用内容高度 + 额外空间）
        pdf_height_inches = max(11.69, (content_height + 100) / 72)  # 72 DPI
        print(f'  PDF纸张高度: {pdf_height_inches:.2f} 英寸')
        
        # 保存为PDF
        print_options = {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'preferCSSPageSize': False,
            'paperWidth': 8.27,
            'paperHeight': pdf_height_inches,
            'marginTop': 0.2,
            'marginBottom': 0.2,
            'marginLeft': 0.3,
            'marginRight': 0.3,
            'scale': 1.0,
        }
        
        result = driver.execute_cdp_cmd('Page.printToPDF', print_options)
        
        # 保存PDF
        import base64
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(result['data']))
        
        print(f'  ✓ 已保存: {filepath}')
        return True
        
    except Exception as e:
        print(f'  ✗ 保存失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程序"""
    # 确定目标日期
    if TARGET_DATE:
        target_date = TARGET_DATE
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    print('='*80)
    print(f'开始处理期货研报邮件 - 目标日期: {target_date}')
    print('='*80)

    try:
        # 加载已有文件列表用于去重
        existing_ids = get_existing_report_ids()
        if existing_ids:
            print(f'\n已有 {len(existing_ids)} 篇报告，将跳过重复下载')
        
        print('\n正在搜索邮件...')
        results = helper.search_email(
            folder=EMAIL_FOLDER,
            pattern=EMAIL_PATTERN,
            start=0,
            limit=EMAIL_SEARCH_LIMIT
        )
        
        if not results:
            print('未找到邮件')
            return
        
        print(f'找到 {len(results)} 封邮件\n')
        
        print('正在启动浏览器...')
        driver = webdriver.Chrome(options=chrome_options)
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        date_skip_count = 0  # 跳过的非目标日期邮件数
        consecutive_other_date = 0  # 连续非目标日期计数
        
        for idx, email in enumerate(results, 1):
            print(f'\n[{idx}/{len(results)}] 处理邮件...')
            
            try:
                if isinstance(email, dict):
                    email_id = email.get('id')
                    # 获取邮件接收日期
                    email_date = email.get('receivedDate') or email.get('sentDate')
                else:
                    email_id = email
                    email_date = None
                
                # 如果没有从搜索结果中获取日期，尝试从邮件详情中获取
                if not email_date:
                    email_info = helper.get_email_info(email_id)
                    if email_info:
                        email_date = email_info.get('receivedDate') or email_info.get('sentDate')
                
                # 如果没有日期，使用当天日期
                if not email_date:
                    email_date = datetime.now().strftime('%Y-%m-%d')
                    print(f'  警告: 无法获取邮件日期，使用当天日期: {email_date}')
                
                # 解析邮件日期，只取日期部分
                if 'T' in email_date:
                    email_date_part = email_date.split('T')[0]
                else:
                    email_date_part = email_date[:10]
                
                # 检查是否为目标日期
                if email_date_part != target_date:
                    print(f'  ⊘ 非目标日期({email_date_part})，跳过')
                    date_skip_count += 1
                    consecutive_other_date += 1
                    # 如果连续遇10封非目标日期的邮件，提前结束（邮件按日期降序）
                    if consecutive_other_date >= 10:
                        print(f'\n连续{consecutive_other_date}封邮件非目标日期，提前结束搜索')
                        break
                    continue
                
                # 重置连续计数
                consecutive_other_date = 0
                
                # 获取日期子文件夹
                save_dir = get_date_subfolder(email_date)
                
                link = extract_link_from_email(email_id)
                if not link:
                    print('  ✗ 未找到链接')
                    fail_count += 1
                    continue
                
                match = re.search(r'id=(\d+)&taskId=(\d+)', link)
                if match:
                    report_key = f'{match.group(2)}_{match.group(1)}'
                    filename = f'期货观点_{report_key}'
                else:
                    report_key = None
                    filename = f'期货观点_{idx}'
                
                # 去重检查：如果已下载则跳过
                if report_key and report_key in existing_ids:
                    print(f'  ⊘ 已存在，跳过: {filename}')
                    skip_count += 1
                    continue
                
                if save_page_as_pdf(driver, link, filename, save_dir, email_date_part):
                    success_count += 1
                    if report_key:
                        existing_ids.add(report_key)
                else:
                    fail_count += 1
                
                time.sleep(2)
                
            except Exception as e:
                print(f'  ✗ 处理失败: {str(e)}')
                fail_count += 1
        
        driver.quit()
        print('\n浏览器已关闭')
        
        print('\n' + '='*80)
        print(f'处理完成！')
        print(f'  成功: {success_count} 个')
        print(f'  跳过(已存在): {skip_count} 个')
        print(f'  跳过(非目标日期): {date_skip_count} 个')
        print(f'  失败: {fail_count} 个')
        print(f'  保存位置: {BASE_SAVE_DIR}')
        print('='*80)
        
    except Exception as e:
        print(f'\n发生错误: {str(e)}')
        import traceback
        traceback.print_exc()
        try:
            driver.quit()
        except:
            pass


if __name__ == '__main__':
    main()
