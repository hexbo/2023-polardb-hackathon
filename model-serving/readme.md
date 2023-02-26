# flask-ai-service 使用说明
该测试程序仅用于模型快速开发验证，因此 server.py 程序中用了同步的方式，在生产环境中需要修改为异步方式，增加消息队列。
```py
app.config["PERMANENT_SESSION_LIFETIME"] = None # 同步处理，不超时
```

# 接口
```
POST /api/query-by-binary
POST /api/predict
```