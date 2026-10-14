"""engine 包：语音链路的"可替换引擎库"，由 ws_server.load_engines() 装配、conversation_manager 编排。

组件之间互不 import，只通过统一的方法签名通信；本文件刻意不写代码，避免 import engine 就拉入重依赖。
"""
