"""
下载期货研报（完整版/详细版）
从Coremail邮件系统下载期货观点邮件，通过"点击查看"获取完整版报告并保存为PDF

使用方法:
1. 复制 param.json.example 为 param.json 并填入邮箱账号密码
2. 安装依赖: pip install selenium requests
3. 运行: python download_detailed_reports.py

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

# ========================================
# 只禁用 Python 与 ChromeDriver 之间的代理
# 但保留 Chrome 浏览器的代理（用于访问外部网站）
# ========================================

# 保存原始代理设置供 Chrome 使用
_saved_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

# 导入 urllib3 并在 Selenium 导入前 patch
import urllib3
urllib3.disable_warnings()

# 保存原始的 PoolManager
_OrigPoolManager = urllib3.PoolManager

class LocalhostNoProxyPoolManager(_OrigPoolManager):
    """PoolManager that bypasses proxy for localhost connections only"""
    def urlopen(self, method, url, **kw):
        # 对于 localhost 连接，不使用代理
        return super().urlopen(method, url, **kw)

# 确保只绕过 localhost 和内部服务器的代理
os.environ['NO_PROXY'] = '127.0.0.1,localhost,*.cicc.group,mailbj.cicc.group'
os.environ['no_proxy'] = '127.0.0.1,localhost,*.cicc.group,mailbj.cicc.group'

from coremail_helper import CoremailHelper
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import re
import glob as glob_module

# Patch Selenium 的 RemoteConnection 使用无代理的 PoolManager 连接 ChromeDriver
from selenium.webdriver.remote import remote_connection as rc

# 保存原始的 RemoteConnection._conn 初始化
_orig_init = rc.RemoteConnection.__init__

def _patched_init(self, *args, **kwargs):
    """Patched init that forces no proxy for local ChromeDriver connection"""
    # 强制忽略代理（因为是连接本地 ChromeDriver）
    kwargs['ignore_proxy'] = True
    _orig_init(self, *args, **kwargs)

rc.RemoteConnection.__init__ = _patched_init

from datetime import datetime

# ============ 配置区域 ============
# 保存目录（可根据需要修改）
BASE_SAVE_DIR = r'D:\QIS_DASHBOARD\memory\commodity_detailed_reports'

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
    
    subfolder_name = f'{date_part} detailed reports'
    subfolder_path = os.path.join(BASE_SAVE_DIR, subfolder_name)
    
    # 创建子文件夹（如果不存在）
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        print(f'创建日期子文件夹: {subfolder_path}')
    
    return subfolder_path

# 设置Chrome选项
chrome_options = Options()
chrome_options.add_argument('--headless=new')  # 使用新版 headless 模式，更好的JS支持
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--window-size=1920,1080')  # 设置窗口大小
chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
chrome_options.add_argument('--disable-blink-features=AutomationControlled')  # 隐藏自动化特征
chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
chrome_options.add_experimental_option('useAutomationExtension', False)
# 设置 Chrome 代理（用于访问外部网站）
if _saved_proxy:
    chrome_options.add_argument(f'--proxy-server={_saved_proxy}')


def get_existing_report_ids():
    """扫描所有日期子文件夹中已下载的PDF文件，提取已有的 taskId_id 组合，用于去重"""
    existing = set()
    if os.path.exists(BASE_SAVE_DIR):
        # 扫描所有子文件夹（日期文件夹）
        for subfolder in os.listdir(BASE_SAVE_DIR):
            subfolder_path = os.path.join(BASE_SAVE_DIR, subfolder)
            if os.path.isdir(subfolder_path):
                for f in glob_module.glob(os.path.join(subfolder_path, '期货详细观点_*.pdf')):
                    basename = os.path.basename(f)
                    # 文件名格式: 期货详细观点_{taskId}_{id}_详细_{title}.pdf
                    match = re.match(r'期货详细观点_(\d+_\d+)', basename)
                    if match:
                        existing.add(match.group(1))
        # 也扫描根目录（兼容旧文件）
        for f in glob_module.glob(os.path.join(BASE_SAVE_DIR, '期货详细观点_*.pdf')):
            basename = os.path.basename(f)
            match = re.match(r'期货详细观点_(\d+_\d+)', basename)
            if match:
                existing.add(match.group(1))
    return existing


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


def get_full_report_link(driver, summary_url):
    """从概要页面获取完整报告链接"""
    try:
        driver.get(summary_url)
        time.sleep(5)
        
        # 查找"点击查看"链接
        link_element = driver.find_element(By.XPATH, "//a[contains(text(), '点击查看')]")
        full_report_link = link_element.get_attribute('href')
        # 打印完整链接用于调试
        print(f'  完整报告链接: {full_report_link}')
        return full_report_link
    except Exception as e:
        print(f'  ✗ 获取完整报告链接失败: {str(e)}')
        return None


def save_full_report_as_pdf(driver, url, filename, save_dir):
    """访问完整报告并保存为PDF
    
    Args:
        driver: Selenium WebDriver实例
        url: 完整报告的URL
        filename: 文件名前缀
        save_dir: 保存目录
    """
    try:
        print(f'  正在访问完整报告: {url[:80]}...')
        driver.get(url)
        time.sleep(3)
        
        # 等待页面内容加载完成（最多等待30秒）
        max_wait = 30
        wait_interval = 2
        waited = 0
        page_title = ""
        
        while waited < max_wait:
            page_title = driver.title
            # 检查是否有有效标题（包含"期货"或"研报"关键字）
            if page_title and ('期货' in page_title or '研报' in page_title or '观点' in page_title):
                print(f'  页面标题: {page_title}')
                break
            
            # 检查页面是否有实际内容（排除 "Alice" 占位符页面）
            body_text = driver.execute_script("return document.body ? document.body.innerText : ''")
            # 跳过 "Alice" 占位符页面
            if body_text and body_text.strip().startswith('Alice'):
                time.sleep(wait_interval)
                waited += wait_interval
                print(f'  等待页面加载... ({waited}s) [检测到占位符]')
                continue
            
            if body_text and len(body_text) > 500 and ('研报' in body_text or '期货' in body_text or '观点' in body_text):
                print(f'  页面标题: {page_title if page_title else "(无标题但有内容)"}')
                break
            
            time.sleep(wait_interval)
            waited += wait_interval
            print(f'  等待页面加载... ({waited}s)')
        
        # 如果等待超时且没有有效内容，跳过此页面
        if waited >= max_wait:
            body_text = driver.execute_script("return document.body ? document.body.innerText : ''")
            # 打印页面内容用于调试
            print(f'  调试: 页面内容前200字符: {body_text[:200] if body_text else "(空)"}')
            # 检测 "Alice" 占位符页面
            if body_text and body_text.strip().startswith('Alice'):
                print(f'  ✗ Wind页面未加载（显示Alice占位符），可能需要登录或链接已过期')
                return False
            if not body_text or len(body_text) < 500:
                print(f'  ✗ 页面内容为空或加载超时，跳过')
                return False
            print(f'  警告: 页面加载超时，但检测到内容，继续保存')
        
        # 再次获取标题（可能已更新）
        page_title = driver.title
        
        # 查找并滚动可滚动容器
        print(f'  正在加载完整页面内容...')
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
            print(f'  找到可滚动容器: {scroll_result["scrollHeight"]}px')
            content_height = scroll_result['scrollHeight']
            
            # 滚动容器
            scroll_container = driver.execute_script("""
                var elements = Array.from(document.querySelectorAll('*'));
                var scrollable = elements.filter(el => el.scrollHeight > el.clientHeight && el.scrollHeight > 500);
                scrollable.sort((a, b) => b.scrollHeight - a.scrollHeight);
                return scrollable[0];
            """)
            
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
            
            driver.execute_script("arguments[0].scrollTop = 0", scroll_container)
            time.sleep(1)
            print(f'  滚动完成（{scroll_attempts + 1}次）')
        else:
            # 如果没有找到特定的滚动容器，尝试滚动整个页面
            content_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
            print(f'  页面高度: {content_height}px')
            
            last_height = content_height
            scroll_attempts = 0
            
            while scroll_attempts < 10:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.body.scrollHeight")
                
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1
            
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
        
        # 最终检查：如果页面标题为空，说明内容未加载成功，跳过保存
        if not page_title or not page_title.strip():
            body_text = driver.execute_script("return document.body ? document.body.innerText : ''")
            if not body_text or len(body_text) < 500 or ('期货' not in body_text and '研报' not in body_text):
                print(f'  ✗ 页面无有效内容（无标题、内容少于500字符），跳过')
                return False
            print(f'  警告: 页面无标题但有内容，继续保存')
        
        # 清理文件名
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', page_title)
        if safe_title:
            filename = f'{filename}_详细_{safe_title}.pdf'
        else:
            filename = f'{filename}_详细.pdf'
        
        filepath = os.path.join(save_dir, filename)
        
        # 计算PDF高度
        pdf_height_inches = max(11.69, (content_height + 100) / 72)
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
    print(f'开始下载期货详细观点报告 - 目标日期: {target_date}')
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
                
                # 获取概要页面链接
                summary_link = extract_link_from_email(email_id)
                if not summary_link:
                    print('  ✗ 未找到概要链接')
                    fail_count += 1
                    continue
                
                # 生成文件名并检查去重
                match = re.search(r'id=(\d+)&taskId=(\d+)', summary_link)
                if match:
                    report_key = f'{match.group(2)}_{match.group(1)}'
                    filename = f'期货详细观点_{report_key}'
                else:
                    report_key = None
                    filename = f'期货详细观点_{idx}'
                
                # 去重检查：如果已下载则跳过
                if report_key and report_key in existing_ids:
                    print(f'  ⊘ 已存在，跳过: {filename}')
                    skip_count += 1
                    continue
                
                # 从概要页面获取完整报告链接
                full_report_link = get_full_report_link(driver, summary_link)
                if not full_report_link:
                    print('  ✗ 未找到完整报告链接')
                    fail_count += 1
                    continue
                
                print(f'  找到完整报告链接')
                
                # 下载完整报告为PDF
                if save_full_report_as_pdf(driver, full_report_link, filename, save_dir):
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
