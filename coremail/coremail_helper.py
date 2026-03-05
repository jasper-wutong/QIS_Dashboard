"""
Coremail 邮件系统高级助手类
提供邮件搜索、发送、下载等功能的封装
"""

from coremail_utils import CoremailHandler
import json
import os
import re

# 获取当前文件所在目录
path = os.path.dirname(os.path.realpath(__file__))

# 尝试加载配置文件
param = None
param_file = os.path.join(path, 'param.json')
if os.path.exists(param_file):
    with open(param_file, encoding='utf-8') as f:
        try:
            text = f.read()
            param = json.loads(text)
        except Exception as e:
            print(f'Warning: Failed to load param.json: {e}')


class CoremailHelper:
    def __init__(self, uid=None, password=None, testing=False, user=None):
        """
        初始化 CoremailHelper
        
        Args:
            uid: 邮箱地址，如果为None则从param.json读取
            password: 邮箱密码，如果为None则从param.json读取
            testing: 是否使用测试环境
            user: 使用哪个用户配置 ('me' 或 'pricing')
        """
        if uid is None:
            if param is None:
                raise ValueError("请提供uid和password，或创建param.json配置文件")
            if user == 'pricing':
                uid = param['users']['pricing']['uid']
                password = param['users']['pricing']['password']
            else:
                uid = param['users']['me']['uid']
                password = param['users']['me']['password']

        url = 'https://mailbj.cicc.group/coremail/s/json?func='
        self.domain = 'cicc.com.cn'
        if testing:
            url = 'https://bjmail.cicccs.group/coremail/s/json?func='
            self.domain = 'cs.cicc.com.cn'
        self.__handler = CoremailHandler(url, uid, password)

    def get_email_addresses(self, addrStr=''):
        """解析邮箱地址字符串"""
        if addrStr:
            addr_list = addrStr.split(';')
            for i in range(0, len(addr_list)):
                if '@' not in addr_list[i]:
                    addr_list[i] = addr_list[i] + '@' + self.domain
            return addr_list
        return []

    def send_mail(self, recipients='', cc='', body='', behalf='', subject='', 
                  attachments=None, auto=False, attrs=None, bcc='', signed=False):
        """
        发送邮件
        
        Args:
            recipients: 收件人（分号分隔）
            cc: 抄送（分号分隔）
            body: 邮件正文（HTML格式）
            behalf: 代发账号
            subject: 邮件主题
            attachments: 附件列表
            auto: 是否自动发送（True=发送，False=保存草稿）
            attrs: 额外属性
            bcc: 密送
            signed: 是否添加签名
        """
        if signed:
            sign_html = '''<br><br><br>
            Best,<br><br>
            Tong Wu<br>
            中国国际金融股份有限公司 股票业务部<br>
            Email: Tong6.Wu@cicc.com.cn<br>
            地址：北京市朝阳区建外大街1号 国贸写字楼2座6层
            '''
            body += sign_html
        
        compose_id = self.__handler.prepareCompose()
        data = {
            'id': compose_id,
            'attrs': {
                'to': recipients.split(';'),
                'cc': cc.split(';'),
                'bcc': bcc.split(';'),
                'subject': subject,
                'content': body,
                'isHtml': True
            },
            'returnInfo': True
        }
        
        if attrs:
            for key, value in attrs.items():
                data['attrs'][key] = value
        
        if behalf:
            data['attrs']['account'] = behalf
            data['attrs']['from'] = behalf
        
        if attachments is not None and len(attachments) > 0:
            attachments_payload = []
            for attachment in attachments:
                attachment_res = self.__handler.prepareAttachment(compose_id, attachment)
                upload_res = self.__handler.uploadAttachment(compose_id, attachment_res, attachment)
                attachments_payload.append({
                    'id': upload_res['attachmentId'],
                    'type': 'upload',
                    'name': upload_res['fileName']
                })
            data['attachments'] = attachments_payload
        
        if auto is True:
            data['action'] = 'deliver'
        else:
            data['action'] = 'save'

        return self.__handler.sendEmail(data)

    def search_email(self, folder='收件箱', start=0, limit=10, summaryWindowSize=10, 
                     pattern='', conditions=[]):
        """
        搜索邮件
        
        Args:
            folder: 邮件文件夹（如 '收件箱'、'收件箱.Research'、'收件箱.期货研报'）
            start: 起始位置
            limit: 返回数量限制
            summaryWindowSize: 摘要窗口大小
            pattern: 搜索关键字
            conditions: 搜索条件
            
        Returns:
            邮件列表
        """
        return self.__handler.searchEmailList(
            folder=folder,
            start=start,
            limit=limit,
            summaryWindowSize=summaryWindowSize,
            pattern=pattern,
            conditions=conditions
        )

    def get_email_info(self, id):
        """获取邮件详细信息"""
        mail_id = id
        if isinstance(id, dict):
            mail_id = id['id']
        return self.__handler.readMessage(mail_id)

    def download_email(self, id, folder):
        """下载整封邮件为.eml文件"""
        mail_id = id
        if isinstance(id, dict):
            mail_id = id['id']
        email_info = self.__handler.readMessage(mail_id)
        return self.__handler.getMessageData(id, 0, f'{folder}/{email_info["subject"]}.eml')

    def download_email_all_attachments(self, id, folder):
        """下载邮件的所有附件"""
        mail_id = id
        if isinstance(id, dict):
            mail_id = id['id']
        return self.__handler.getMessageAttachementsAll(mail_id, folder)

    def download_email_content(self, id, folder):
        """下载邮件正文为HTML文件"""
        mail_id = id
        if isinstance(id, dict):
            mail_id = id['id']
        email_info = self.__handler.readMessage(mail_id)
        item = email_info['html']
        return self.__handler.getMessagePartialAttachement(
            mail_id, item['id'], f'{folder}/{email_info["subject"]}.html'
        )

    def download_email_attachment(self, id, attachmentId, folder):
        """下载邮件的指定附件"""
        mail_id = id
        if isinstance(id, dict):
            mail_id = id['id']
        email_info = self.__handler.readMessage(mail_id)
        if email_info.get('attachments') and len(email_info['attachments']) > 0:
            for attachment in email_info['attachments']:
                if attachment['id'] == attachmentId:
                    return self.__handler.getMessagePartialAttachement(
                        mail_id, attachment['id'], f'{folder}/{attachment["filename"]}'
                    )

    def get_email_html(self, id):
        """获取邮件HTML内容"""
        mail_id = id
        if isinstance(id, dict):
            mail_id = id['id']
        return self.__handler._CoremailHandler__handler.getHTML(mail_id) if hasattr(self, '_CoremailHelper__handler') else self.__handler.getHTML(mail_id)

    def move_email_to_folder(self, folder, email_ids=[]):
        """移动邮件到指定文件夹"""
        if len(email_ids) > 0 and folder is not None:
            ids = []
            for eid in email_ids:
                if isinstance(eid, dict):
                    ids.append(eid.get('id'))
                else:
                    ids.append(eid)
            folder_id = self.__handler.getFolderIdByName(folder)
            if folder_id is not None:
                self.__handler._req(
                    url='mbox:updateMessageInfos',
                    data={'ids': ids, 'attrs': {'fid': folder_id}}
                )

    def get_email_list(self, arr=''):
        """从字符串中提取邮箱地址列表"""
        matches = []
        pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}')
        if arr:
            matches = pattern.findall(arr)
        return matches

    def reply_email(self, email_id, body='', behalf='', attachments=None, auto=False, 
                    attrs=None, recipients='', cc='', bcc='', add_to='', add_cc='', 
                    subject='', signed=False):
        """
        回复邮件
        
        Args:
            email_id: 要回复的邮件ID
            body: 回复正文
            behalf: 代发账号
            attachments: 附件列表
            auto: 是否自动发送
            attrs: 额外属性
            recipients: 覆盖收件人
            cc: 覆盖抄送
            bcc: 密送
            add_to: 添加收件人
            add_cc: 添加抄送
            subject: 覆盖主题
            signed: 是否添加签名
        """
        if signed:
            sign_html = '''<br><br><br>
            Best,<br><br>
            Tong Wu<br>
            中国国际金融股份有限公司 股票业务部<br>
            Email: Tong6.Wu@cicc.com.cn<br>
            地址：北京市朝阳区建外大街1号 国贸写字楼2座6层
            '''
            body += sign_html

        mail_id = email_id
        if isinstance(email_id, dict):
            mail_id = email_id['id']
        
        data = {
            'id': mail_id,
            'toAll': True,
            'attrs': {
                'isHtml': True
            }
        }
        
        compose_info = self.__handler._req('mbox:replyMessage', data)
        compose_id = ''
        ori_body = ''
        to = []
        email_cc = []
        
        if isinstance(compose_info, dict):
            compose_id = compose_info['id']
            ori_body = compose_info['content']
            to = compose_info['to']
            if 'cc' in compose_info.keys():
                email_cc = compose_info['cc']
        
        data['id'] = compose_id
        data['attrs']['content'] = body + '<br/>' + ori_body
        
        if attrs:
            for key, value in attrs.items():
                data['attrs'][key] = value
        
        if behalf:
            data['attrs']['account'] = behalf
            data['attrs']['from'] = behalf
        
        if attachments is not None and len(attachments) > 0:
            attachments_payload = []
            for attachment in attachments:
                attachment_res = self.__handler.prepareAttachment(compose_id, attachment)
                upload_res = self.__handler.uploadAttachment(compose_id, attachment_res, attachment)
                attachments_payload.append({
                    'id': upload_res['attachmentId'],
                    'type': 'upload',
                    'name': upload_res['fileName']
                })
            data['attachments'] = attachments_payload
        
        if auto is True:
            data['action'] = 'deliver'
        else:
            data['action'] = 'save'
        
        if recipients:
            to = self.get_email_list(recipients)
        if cc:
            email_cc = self.get_email_list(cc)
        if bcc:
            data['attrs']['bcc'] = self.get_email_list(bcc)
        if add_to:
            to.extend(self.get_email_list(add_to))
        if add_cc:
            email_cc.extend(self.get_email_list(add_cc))
        if subject:
            data['attrs']['subject'] = subject
        
        data['attrs']['to'] = to
        data['attrs']['cc'] = email_cc

        return self.__handler.sendEmail(data)
