"""
每日自动上传脚本

功能：
- 自动上传当日下载的 general reports 和 detailed reports
- 用于定时任务调用

使用方式：
    # 上传当天的文件
    python upload_daily.py
    
    # 上传指定日期的文件
    python upload_daily.py --date 2026-02-12
    
    # 只列出文件，不实际上传
    python upload_daily.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 设置输出编码为 UTF-8，避免 Windows 批处理中的 GBK 编码错误
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from upload_manager import SmartUploadManager, log, SUPPORTED_EXTENSIONS


def get_config():
    """获取配置"""
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    
    return {
        'access_key_id': os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        'access_key_secret': os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        'workspace_id': os.getenv('WORKSPACE_ID'),
        'index_id': os.getenv('INDEX_ID'),
    }


def count_files(folder: Path) -> int:
    """统计文件夹中支持的文件数量"""
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="每日自动上传脚本")
    parser.add_argument('--date', '-d', help='目标日期 (默认: 今天)', 
                        default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--dry-run', action='store_true', help='只列出文件，不实际上传')
    parser.add_argument('--skip-parse', action='store_true', help='不等待解析完成')
    parser.add_argument('--skip-index', action='store_true', help='不等待索引完成')
    args = parser.parse_args()
    
    target_date = args.date
    memory_dir = Path(__file__).parent.parent / 'memory'
    
    # 构建当日文件夹路径
    general_folder = memory_dir / 'commodity_general_reports' / f'{target_date} general reports'
    detailed_folder = memory_dir / 'commodity_detailed_reports' / f'{target_date} detailed reports'
    
    log(f"=" * 60)
    log(f"📅 每日上传任务 - 目标日期: {target_date}")
    log(f"=" * 60)
    
    # 检查文件夹
    general_count = count_files(general_folder)
    detailed_count = count_files(detailed_folder)
    
    log(f"\n📁 General Reports: {general_folder}")
    log(f"   文件数: {general_count}")
    
    log(f"\n📁 Detailed Reports: {detailed_folder}")
    log(f"   文件数: {detailed_count}")
    
    total = general_count + detailed_count
    if total == 0:
        log(f"\n⚠️ 没有找到 {target_date} 的报告文件")
        return
    
    log(f"\n📊 总计: {total} 个文件待上传")
    
    if args.dry_run:
        log("\n🔍 [试运行模式] 以下文件将被上传:")
        
        for folder, name in [(general_folder, 'General'), (detailed_folder, 'Detailed')]:
            if folder.exists():
                files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
                if files:
                    log(f"\n  {name} Reports ({len(files)}):")
                    for f in files[:10]:  # 只显示前10个
                        log(f"    - {f.name}")
                    if len(files) > 10:
                        log(f"    ... 还有 {len(files) - 10} 个文件")
        return
    
    # 初始化上传管理器
    config = get_config()
    missing = [k for k, v in config.items() if not v]
    if missing:
        log(f"❌ 缺少配置: {', '.join(missing)}", "ERROR")
        sys.exit(1)
    
    manager = SmartUploadManager(**config)
    
    results_total = {
        'uploaded': [],
        'skipped': [],
        'failed': [],
    }
    
    # 上传 General Reports
    if general_folder.exists() and general_count > 0:
        log(f"\n{'='*40}")
        log(f"📤 开始上传 General Reports...")
        log(f"{'='*40}")
        
        results = manager.sync_folder(
            general_folder,
            wait_parse=not args.skip_parse,
            wait_index=not args.skip_index,
        )
        
        results_total['uploaded'].extend(results['uploaded'])
        results_total['skipped'].extend(results['skipped'])
        results_total['failed'].extend(results['failed'])
    
    # 上传 Detailed Reports (暂时禁用)
    # if detailed_folder.exists() and detailed_count > 0:
    #     log(f"\n{'='*40}")
    #     log(f"📤 开始上传 Detailed Reports...")
    #     log(f"{'='*40}")
    #     
    #     results = manager.sync_folder(
    #         detailed_folder,
    #         wait_parse=not args.skip_parse,
    #         wait_index=not args.skip_index,
    #     )
    #     
    #     results_total['uploaded'].extend(results['uploaded'])
    #     results_total['skipped'].extend(results['skipped'])
    #     results_total['failed'].extend(results['failed'])
    
    # 最终汇总
    log(f"\n{'='*60}")
    log(f"🎉 每日上传任务完成！")
    log(f"📅 日期: {target_date}")
    log(f"📊 结果:")
    log(f"   ✅ 新上传: {len(results_total['uploaded'])}")
    log(f"   ⏭️ 已跳过: {len(results_total['skipped'])}")
    log(f"   ❌ 失败: {len(results_total['failed'])}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
