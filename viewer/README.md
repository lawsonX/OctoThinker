# Parquet Viewer

本地实时查看 parquet 数据的 Web 工具:左边文件树, 右边表格, 打开浏览器即用.

## 启动

```bash
cd /hdd2/xiaolirui/OctoThinker/viewer
./run.sh                 # 默认端口 8790, 默认数据目录 /hdd2/xiaolirui/octoThinker-Datasets
./run.sh 9000 /其它/数据目录   # 可换端口和数据目录
```

然后浏览器打开 `http://<机器IP>:8790`(本机用 `http://127.0.0.1:8790`)。

后台常驻运行: `nohup ./run.sh > viewer.log 2>&1 &`

## 功能

- **点文件**看单个 parquet; **点目录**自动合并该目录下所有分片一起查
- 文件树每 10 秒自动扫描, 新文件标绿色 NEW, 有更新标橙色 UPD(右下角弹提示)
- 点列名排序, 顶部输入框写过滤条件(标准 SQL WHERE, 例 `language = 'Chinese'`), 回车即查
- 「随机抽样」按行随机抽一页, 适合大表快速抽查
- 「SQL 模式」里可对当前文件/目录写任意 SELECT(如 GROUP BY 统计), Ctrl+Enter 运行
- 点列名标签(芯片)看该列统计: 空值数、去重数、min/max/avg
- 点单元格弹窗看完整内容(长文本、messages 等嵌套 JSON 自动格式化)
- 长内容自动截断, 防止单行几十 MB 的卡死

## 技术

后端 FastAPI + DuckDB(过滤/排序/分页全部下推, 只读 SELECT, 带关键字防护),
前端单文件零依赖, 整页离线可用。依赖见 requirements.txt(venv 由 run.sh 自动创建)。
