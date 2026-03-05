"""
Coremail 邮件系统底层处理器
用于与CICC Coremail API交互
"""

import os.path
import requests
import json
import time
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

token_valid_threshold = 3500


class CoremailHandler:
    def __init__(self, url, uid, password):
        self.sid = None
        self.session = None
        self.cookies = None
        self.base_url = url
        self.login_success = False
        self.login_time = None
        self._uid = uid
        self._pass = password

    def login(self):
        now = int(time.time())
        if not self.login_time or now > self.login_time + token_valid_threshold:
            login_url = f'{self.base_url}user:login'
            data = json.dumps({"uid": self._uid, "password": self._pass})
            headers = {'Content-Type': 'text/x-json'}
            # 禁用代理，直连内部邮件服务器
            response = requests.post(login_url, headers=headers, data=data, verify=False, proxies={'http': None, 'https': None})
            self.cookies = response.cookies
            text = json.loads(response.text)
            if text.get('code') == 'S_OK':
                self.sid = text['var']['sid']
                self.login_success = True
                self.login_time = now
            else:
                print('Attention: Fail to login.')

    def _req(self, url, data, **kwargs):
        self.login()
        params = {
            'headers': {'Content-Type': 'text/x-json'},
            'cookies': self.cookies,
            'verify': False,
            'proxies': {'http': None, 'https': None}  # 禁用代理
        }
        if data is not None:
            params['data'] = json.dumps(data)
        for key, value in kwargs.items():
            params[key] = value
        response = requests.post(f'{self.base_url}{url}&sid={self.sid}', **params)
        text = json.loads(response.text)
        if text.get('code') == 'S_OK':
            if text.get('var'):
                return text['var']
            return None
        else:
            print(f'Attention: Fail to request result:-->{text}, response:-->{response}')
            return []

    def _get(self, url, data, **kwargs):
        self.login()
        params = {
            'cookies': self.cookies,
            'verify': False,
            'proxies': {'http': None, 'https': None}  # 禁用代理
        }
        for key, value in kwargs.items():
            params[key] = value
        response = requests.get(
            f'{self.base_url}{url}&sid={self.sid}',
            params=data,
            **params
        )
        text = json.loads(response.text)
        if text.get('code') == 'S_OK':
            return text['var']
        else:
            print(f'Attention: Fail to request result:-->{text}, response:-->{response}')
            return None

    def _req_download(self, data, filename):
        self.login()
        response = requests.get(
            f'{self.base_url}mbox:getMessageData&sid={self.sid}',
            data,
            cookies=self.cookies,
            stream=True,
            verify=False,
            proxies={'http': None, 'https': None}  # 禁用代理
        )
        if response.status_code == 200:
            if filename:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f'Save File to:-->{filename}')
                return True
            else:
                return response.text
        else:
            print(f'Attention: Fail to request result:-->{response.text}, response:-->{response}')
            return False

    def getAllFolders(self):
        url = 'mbox:getAllFolders'
        res = self._req(url, {'flush': True, 'stats': True, 'threads': True})
        return res

    def _deep_search_fold(self, item, name):
        if isinstance(item, list):
            for x in item:
                if x['name'] == name:
                    return x['id'], x.get('children')
            return None, None
        if item['name'] == name:
            return item['id'], item.get('children')
        return None, None

    def _recFolder(self, item, name):
        names = name.split('.')
        for i in range(len(names)):
            folder_id, folder_chidren = self._deep_search_fold(item, names[i])
            item = folder_chidren
        return folder_id

    def getFolderIdByName(self, name):
        if name == '收件箱':
            return 1
        else:
            all_folders = self.getAllFolders()
            folder_id = None
            for item in all_folders:
                if item['name'] == name.split('.')[0]:
                    folder_id = self._recFolder(item, name)
            return folder_id

    def listMessages(self, fid, received_date=None, start=0, limit=-1, summaryWindowSize=0):
        url = 'mbox:listMessages'
        params = {
            "fid": fid,
            "start": start,
            "limit": limit,
            "summaryWindowSize": summaryWindowSize,
            "returnTotal": 'true',
            'order': 'date',
            'desc': True
        }
        if received_date:
            params.update({'filter': {'receivedDate': received_date}})
        res = self._req(url, params)
        return res

    def readMessage(self, id):
        url = 'mbox:readMessage'
        res = self._req(url, {"id": id, "mode": "both"})
        return res

    def getMessageData(self, id, partId=0, filename=None):
        res = self._req_download({'mid': id, 'part': str(partId)}, filename)
        return res

    def getMessageAttachementsAll(self, id, folderPath):
        message = self.readMessage(id)
        final_res = []
        for item in message['attachments']:
            res = self.getMessageData(id, item['id'], f'{folderPath}/{item["filename"]}')
            final_res.append(res)
        return final_res

    def getMessagePartialAttachement(self, id, partId, fileName):
        return self.getMessageData(id, partId, fileName)

    def getHTML(self, id):
        message = self.readMessage(id)
        item = message['html']
        html = self.getMessageData(id, item['id'])
        return html

    def prepareCompose(self):
        return self._get('mbox:compose', {})

    def prepareAttachment(self, composeId, attachment):
        url = 'upload:prepare'
        return self._req(url, {
            "composeId": composeId,
            "fileName": os.path.basename(attachment),
            "size": 10
        })

    def uploadAttachment(self, composeId, attachmentRes, attachment):
        file_name = os.path.basename(attachment)
        attachment_id = attachmentRes['attachmentId']
        url = f'upload:directData&composeId={composeId}&attachmentId={attachment_id}'
        return self._req(url, data=None, files={'file': (file_name, open(attachment, 'rb'))}, headers={})

    def sendEmail(self, data):
        url = 'mbox:compose'
        result = self._req(url, data)
        print('email sent successfully')
        return result

    def searchEmailList(self, folder='收件箱', start=0, limit=10, summaryWindowSize=10, pattern='', conditions=[]):
        fid = self.getFolderIdByName(name=folder)
        if pattern:
            data = {
                'fid': fid,
                'start': start,
                'limit': limit,
                'pattern': pattern,
                'summaryWindowSize': summaryWindowSize,
                'order': 'date',
                'desc': True,
                'conditions': conditions,
                'groupings': {}
            }
            return self._req('mbox:searchMessages', data)
        else:
            return self.listMessages(fid, start=start, limit=limit, summaryWindowSize=summaryWindowSize)
