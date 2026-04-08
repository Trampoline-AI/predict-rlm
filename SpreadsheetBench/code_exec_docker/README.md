## Code Execution Docker

### 修改config
修改code_execution文件夹中的config.json文件，volumes_path字段表示docker中挂载的文件夹路径，请使用绝对路径。
例如现在测试dataset_cell_50，请设置路径为: YOUR_PATH/dataset_cell_50

### 下载Docker镜像
```
cd code_execution
docker build -t xingyaoww/codeact-execute-api -f Dockerfile.api .
docker build -t xingyaoww/codeact-executor -f Dockerfile.executor .
```

### 运行Docker
```
bash start_docker.sh
```

### 测试是否成功
设置 volumes_path 为 YOUR_PATH/code_exec_docker/data
```
cd ..
python test.py
```
